import asyncio
import multiprocessing
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from multiprocessing import Queue
from multiprocessing.connection import wait
from multiprocessing.process import BaseProcess
from typing import (Any, Callable, Dict, Generic, List, Optional, TextIO,
                    TypeVar, Union)

import torch
import zmq
import pickle

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.triton_utils.importing import HAS_TRITON
from vllm.utils import cuda_is_initialized

if HAS_TRITON:
    from vllm.triton_utils import maybe_set_triton_cache_manager

logger = init_logger(__name__)

T = TypeVar('T')

_TERMINATE = "TERMINATE"  # sentinel

# ANSI color codes
CYAN = '\033[1;36m'
RESET = '\033[0;0m'

JOIN_TIMEOUT_S = 2

zmqContext = zmq.Context()

@dataclass
class Result(Generic[T]):
    """Result of task dispatched to worker"""

    task_id: uuid.UUID
    value: Optional[T] = None
    exception: Optional[BaseException] = None


class ResultFuture(threading.Event, Generic[T]):
    """Synchronous future for non-async case"""

    def __init__(self):
        super().__init__()
        self.result: Optional[Result[T]] = None

    def set_result(self, result: Result[T]):
        self.result = result
        self.set()

    def get(self) -> T:
        self.wait()
        assert self.result is not None
        if self.result.exception is not None:
            raise self.result.exception
        return self.result.value  # type: ignore[return-value]


def _set_future_result(future: Union[ResultFuture, asyncio.Future],
                       result: Result):
    if isinstance(future, ResultFuture):
        future.set_result(result)
        return
    loop = future.get_loop()
    if not loop.is_closed():
        if result.exception is not None:
            loop.call_soon_threadsafe(future.set_exception, result.exception)
        else:
            loop.call_soon_threadsafe(future.set_result, result.value)


class ResultHandler(threading.Thread):
    """Handle results from all workers (in background thread)"""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        
        master_addr = os.environ["MASTER_ADDR"]
        self.ZMQ_URL_Result = f"tcp://{master_addr}:{5200}"
        self.result_socket = zmqContext.socket(zmq.PULL)
        self.result_socket.bind(self.ZMQ_URL_Result)
        self.tasks: Dict[uuid.UUID, Union[ResultFuture, asyncio.Future]] = {}

    def run(self):
        while(result := self.result_socket.recv_pyobj()) != _TERMINATE:
            future = self.tasks.pop(result.task_id)
            _set_future_result(future, result)
        # Ensure that all waiters will receive an exception
        for task_id, future in self.tasks.items():
            _set_future_result(
                future,
                Result(task_id=task_id,
                       exception=ChildProcessError("worker died")))

    def close(self):
        print("Sending terminate to result handler")
        result_socket_push = zmqContext.socket(zmq.PUSH)
        result_socket_push.connect(self.ZMQ_URL_Result)
        result_socket_push.send_pyobj(_TERMINATE)
        print("Sent terminate to result handler")

class WorkerMonitor(threading.Thread):
    """Monitor worker status (in background thread)"""

    def __init__(self, workers: List['ProcessWorkerWrapper'],
                 result_handler: ResultHandler):
        super().__init__(daemon=True)
        self.workers = workers
        self.result_handler = result_handler
        self._close = False

    def run(self) -> None:
        while True:
            # TODO
            time.sleep(10)

    def close(self):
        if self._close:
            return
        self._close = True
        logger.info("Terminating local vLLM worker processes")
        for worker in self.workers:
            worker.terminate_worker()
        # Must be done after worker task queues are all closed
        self.result_handler.close()


class ProcessWorkerWrapper:
    """Local process wrapper for vllm.worker.Worker,
    for handling single-node multi-GPU tensor parallel."""

    def __init__(self, rank , result_handler: ResultHandler,
                 worker_factory: Callable[[], Any]) -> None:

        # Connect to worker sockets, worker should already running
        master_addr = os.environ["MASTER_ADDR"]
        self.rank = rank
        ZMQ_URL_Task = f"tcp://{master_addr}:{5100 + rank}"

        self.socket_Task = zmqContext.socket(zmq.PUSH)
        self.socket_Task.bind(ZMQ_URL_Task)
        print(f"WorkerWrapper {rank} connected to Task Queue {ZMQ_URL_Task}")

        # Passing the worker factory to the worker process
        self.socket_Task.send_pyobj(worker_factory)
        print(f"WorkerWrapper {rank} sent worker factory")
        self.tasks = result_handler.tasks

    def _enqueue_task(self, future: Union[ResultFuture, asyncio.Future],
                      method: str, args, kwargs):
        task_id = uuid.uuid4()
        self.tasks[task_id] = future
        try:
            self.socket_Task.send_pyobj((task_id, method, args, kwargs))            
        except Exception as e:
            logger.error(f"Error sending task to worker {self.rank}: {e}")
            raise e

    def execute_method(self, method: str, *args, **kwargs):
        future: ResultFuture = ResultFuture()
        self._enqueue_task(future, method, args, kwargs)
        return future

    async def execute_method_async(self, method: str, *args, **kwargs):
        future = asyncio.get_running_loop().create_future()
        self._enqueue_task(future, method, args, kwargs)
        return await future

    def terminate_worker(self):
        self.socket_Task.send_pyobj(_TERMINATE)
        
    def kill_worker(self):
        self.socket_Task.send_pyobj(_TERMINATE)


def mpi_worker_process() -> None:
    # Queue connecting
    master_addr = os.environ["MASTER_ADDR"]
    rank = int(os.environ['RANK'])
    ZMQ_URL_Task = f"tcp://{master_addr}:{5100 + rank}"
    ZMQ_URL_Result = f"tcp://{master_addr}:{5200}" # result for all workers

    socket_pull = zmqContext.socket(zmq.PULL)
    socket_pull.connect(ZMQ_URL_Task)
    print(f"Worker {rank} bound to Task Queue {ZMQ_URL_Task}")
    socket_push = zmqContext.socket(zmq.PUSH)
    socket_push.connect(ZMQ_URL_Result)
    print(f"Worker {rank} connect to Result Queue {ZMQ_URL_Result}")

    # create worker
    worker_factory: Callable[[], Any] = socket_pull.recv_pyobj()
    print(f"Worker {rank} received worker factory")
    worker = worker_factory()
    del worker_factory
    print(f"Worker {rank} created worker")

    # loop execute
    logger.info(f"Worker {rank} ready; awaiting tasks")
    try:
        while ((items := socket_pull.recv_pyobj()) != _TERMINATE):
            output = None
            exception = None
            print(f"Worker {rank} received task {items}")
            task_id, method, args, kwargs = items
            try:
                executor = getattr(worker, method)
                output = executor(*args, **kwargs)
            except Exception as e:
                exception = e
                logger.exception(f"Worker {rank} task failed: {e}")
            socket_push.send_pyobj(
                Result(task_id=task_id, value=output, exception=exception))

    except KeyboardInterrupt:
        pass
    except Exception:
        logger.exception(f"Worker {rank} failed")

    logger.info(f"Worker {rank} exiting")


def set_multiprocessing_worker_envs(parallel_config):
    """ Set up environment variables that should be used when there are workers
    in a multiprocessing environment. This should be called by the parent 
    process before worker processes are created"""

    if (cuda_is_initialized()
            and os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn"):
        logger.warning("CUDA was previously initialized. We must use "
                       "the `spawn` multiprocessing start method. Setting "
                       "VLLM_WORKER_MULTIPROC_METHOD to 'spawn'.")
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    # Configure thread parallelism if OMP_NUM_THREADS isn't set
    #
    # Helps to avoid CPU contention. The default of spawning a thread per
    # core combined with multiprocessing for each GPU can have a negative
    # impact on performance. The contention is amplified when running in a
    # container where CPU limits can cause throttling.
    default_omp_num_threads = 1
    if "OMP_NUM_THREADS" not in os.environ and (
            current_parallelism :=
            torch.get_num_threads()) > default_omp_num_threads:
        logger.warning(
            "Reducing Torch parallelism from %d threads to %d to avoid "
            "unnecessary CPU contention. Set OMP_NUM_THREADS in the "
            "external environment to tune this value as needed.",
            current_parallelism, default_omp_num_threads)
        os.environ["OMP_NUM_THREADS"] = str(default_omp_num_threads)
        torch.set_num_threads(default_omp_num_threads)

    # workaround for https://github.com/vllm-project/vllm/issues/6103
    if HAS_TRITON and parallel_config.world_size > 1:
        maybe_set_triton_cache_manager()
