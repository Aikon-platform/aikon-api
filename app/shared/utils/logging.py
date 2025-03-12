"""
A module for logging progress
TODO clean all of this, this is very messy
"""

import functools
import json
import logging
import os

import time
from enum import Enum
from pathlib import Path

import dramatiq
from tqdm import tqdm
from dramatiq.middleware import CurrentMessage
from dramatiq.results import Results, ResultBackend
import traceback
from torch import Tensor
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Type,
    TypeVar,
    Union,
)
import requests


T = TypeVar("T")


def notifying(func: Optional[Callable[..., Any]] = None) -> Callable[..., Any]:
    """
    A decorator to notify the task of the progress of a function
    Sends back to frontend the results returned be the task when it is done
    Gives the task a logger and a notifier to log/notify its progress as a kwarg

    Raised exceptions are dispatched as ERROR events to frontend

    Statuses:
    - STARTED indicates that the task has been triggered
    - PROGRESS when intermediary results are dispatched to the frontend but task is not done
    - SUCCESS indicates that the task has completed successfully
    - ERROR indicates that the task has failed because an Exception was raised
            (if notify("ERROR", ..., completed=False) => AIKON doesn't trigger the end of the task)
    """

    @functools.wraps(func)
    def wrapper(fct: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fct)
        def execute(*args, **kwargs):
            logger = JobLogger.getLogger(create=True)
            logger.info(f"Starting task {fct.__name__}")
            current_task_id = getattr(logger, "_id", None)
            notify_url = kwargs.get("notify_url", None)
            experiment_id = kwargs.get("experiment_id", None)

            def notify(event: str, **data):
                if notify_url:
                    requests.post(
                        notify_url,
                        json={
                            "event": event,
                            "tracking_id": current_task_id,
                            "experiment_id": experiment_id,
                            **json.loads(json.dumps(data, default=serializer)),
                        },
                    )

            try:
                notify("STARTED")
                result = fct(*args, **kwargs, logger=logger, notifier=notify)
                # dispatch result to frontend
                notify("SUCCESS", success=True, output=result)

                return result
            except Exception as e:
                logger.error(f"Error in task {fct.__name__}", exception=e)
                try:
                    notify("ERROR", error=traceback.format_exc(), completed=True)
                except Exception as e:
                    logger.error("Error while notifying", exception=e)

        return execute

    return wrapper if func is None else wrapper(func)


def serializer(obj):
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Type {type(obj)} is not JSON serializable")


def exc_str(e: Exception):
    return f"[{e.__class__.__name__}] {e}\nStack Trace:\n{traceback.format_exc()}\n"


def sanitize(v):
    """
    Helper function to convert non-serializable values to string representations.
    """
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    elif isinstance(v, (list, tuple)):
        return [sanitize(x) for x in v]
    elif isinstance(v, dict):
        return {str(k): sanitize(val) for k, val in v.items()}
    else:
        # For custom objects, include class name in representation
        return f"{v.__class__.__name__}({str(v)})"


def pprint(o):
    if isinstance(o, str):
        try:
            return json.dumps(json.loads(o), indent=4, sort_keys=True)
        except ValueError:
            return o
    elif isinstance(o, dict) or isinstance(o, list):
        try:
            return json.dumps(o, indent=4, sort_keys=True)
        except TypeError:
            try:
                if isinstance(o, dict):
                    sanitized = {str(k): sanitize(v) for k, v in o.items()}
                else:
                    sanitized = [sanitize(v) for v in o]
                return json.dumps(sanitized, indent=4, sort_keys=True)
            except Exception:
                return str(o)
    elif isinstance(o, Tensor):
        return pprint(o.tolist())
    return str(o)


class ConsoleColors:
    """
    Color codes for console output

    Last digit:
    0	black
    1	red
    2	green
    3	yellow
    4	blue
    5	magenta
    6	cyan
    7	white
    """

    black = "\033[90m"
    red = "\033[91m"
    green = "\033[92m"
    yellow = "\033[93m"
    blue = "\033[94m"
    magenta = "\033[95m"
    cyan = "\033[96m"
    white = "\033[97m"
    bold = "\033[1m"
    underline = "\033[4m"
    end = "\033[0m"


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def get_color(color=None):
    return getattr(ConsoleColors, color or "blue", "\033[94m")


base_logger = logging.getLogger("DEMO_API")
"""The base logger for the application"""


class TqdmProgress(tqdm):
    """
    A TQDM progress bar that can be used to monitor the progress of an iterable, sending updates to a callback
    """

    def __init__(self, iterable, *args, **kwargs):
        # extract a vc_callback from the kwargs
        self._vc_progress_callback = kwargs.pop("vc_progress_callback", None)
        self._vc_close_callback = kwargs.pop("vc_end_callback", None)
        super().__init__(iterable, *args, **kwargs)

    def update(self, n: int = 1):
        # Overwrite update to call the callback
        displayed = super().update(n)
        if displayed and self._vc_progress_callback is not None:
            self._vc_progress_callback(self)
        return displayed

    def close(self):
        # Overwrite close to call the callback
        closed = super().close()
        if self._vc_close_callback is not None:
            self._vc_close_callback(self)
        return closed


class LoggerHelper:
    """
    A helper static class to log progress and errors
    """

    def __init__(self):
        raise ValueError("This class is not meant to be instanced")

    @staticmethod
    def info(*s, **kwargs) -> None:
        """
        Log an info message
        """
        base_logger.info("\n".join(str(p) for p in s))

    @staticmethod
    def warning(*s, exception: bool = False, **kwargs) -> None:
        """
        Log a warning message

        :param s: The messages to log
        :param exception: Add the exception to the log
        """
        text = "\n".join(str(p) for p in s)

        if exception:
            text += f"\n\n{traceback.format_exc(limit=1)}"

        base_logger.warning(text)

    @staticmethod
    def error(*s, exception: Exception = None, **kwargs):
        """
        Log an error message

        :param s: The messages to log
        :param exception: Add the exception to the log
        """
        text = "\n".join(str(p) for p in s)

        if exception:
            text += f"\n\n{traceback.format_exc()}"

        base_logger.error(text)

    @classmethod
    def progress(
        cls, current: int = 0, total: int = None, title: str = "", **kwargs
    ) -> None:
        """
        Log the progress of a task

        :param current: The current progress
        :param total: The total progress
        :param title: The title of the task
        """
        base_logger.info(f"Progress {title} {current}/{total}")

    @classmethod
    def iterate(
        cls,
        iterable: Iterable[T],
        title: str = "",
        total: Optional[int] = None,
        rate_limit: float = 1.0,
    ) -> Iterable[T]:
        """
        Monitor the progress of iterating an iterable (through tqdm)

        :param iterable: The iterable to monitor
        :param title: The title of the task
        :param total: The total number of items in the iterable
        :param rate_limit: The minimum interval between updates
        """
        cls.progress(0, total, title=title)

        def progress_callback(prog: TqdmProgress):
            cls.progress(prog.n, prog.total, title)

        def end_callback(prog: TqdmProgress):
            base_logger.info(f"End {title}")

        return TqdmProgress(
            iterable,
            vc_progress_callback=progress_callback,
            vc_end_callback=end_callback,
            desc=title,
            mininterval=rate_limit,
            total=total,
        )


TLoggerHelper = Type[LoggerHelper]
TJobLogger = TypeVar("TJobLogger", bound="JobLogger")
TLogger = Union[TJobLogger, TLoggerHelper]


class JobLogger:
    """
    A class to log the progress of a job and send it back to the frontend
    """

    LOGGERS: Dict[str, "JobLogger"] = {}

    def __init__(self, message: dramatiq.Message, description=None):
        self._errors = []
        self._warnings = []
        self._grouped_warnings = {}
        self._progress = {}
        self._spawned = []
        self._forwarded = {}
        self._latest_infos = []
        self._message = message
        self._id = message.message_id
        self.description = description
        self._result_backends = []

    def register_backend(self, backend: ResultBackend) -> None:
        """
        Register a result backend to store the results of the job
        """
        self._result_backends.append(backend)

    @staticmethod
    def getLoggingRedisKey(message_id: str):
        return f"job:{message_id}"

    @classmethod
    def getLogger(cls: Type[TJobLogger], create: bool = False) -> TLogger:
        """
        Return or create the appropriate logger for the job

        Only one logger is created per job, and it is stored in a class variable

        :param create: Whether to create a new logger if it does not exist

        :return: The logger for the job (or the helper if no job is running)
        """
        current_task = CurrentMessage.get_current_message()
        if not current_task:
            return LoggerHelper

        job_id = current_task.message_id

        if create and job_id not in cls.LOGGERS:
            cls.LOGGERS[job_id] = cls(current_task)

        return cls.LOGGERS[job_id]

    @classmethod
    def clearLogger(
        cls: Type[TJobLogger], job_id: Optional[str] = None
    ) -> Optional[TLogger]:
        """
        Clear the logger for the job
        """
        current_task = CurrentMessage.get_current_message()
        if job_id is None:
            if not current_task:
                return None

            job_id = current_task.message_id

        if job_id in cls.LOGGERS:
            return cls.LOGGERS.pop(job_id)

        return None

    def terminate(self) -> Dict[str, str]:
        """
        Terminate the job and return the final state
        """
        self.clearLogger(self._id)
        return self.get_state(with_warnings=True)

    def get_state(self, with_warnings: bool = False) -> Dict[str, Any]:
        """
        Get the state of the current task
        """
        state = {"id": self._id}

        if self.description:
            state["description"] = self.description

        if self._spawned:
            state["spawned"] = list(self._spawned)

        if self._errors:
            state["errors"] = self._errors

        if self._progress:
            state["progress"] = list(self._progress.values())

        if self._forwarded:
            state["forwarded"] = list(self._forwarded.values())

        if self._latest_infos:
            state["infos"] = self._latest_infos

        if with_warnings:
            warnings = self._warnings
            if self._grouped_warnings:
                for collapse, ws in self._grouped_warnings.items():
                    warnings = [
                        f"{len(ws)} {collapse} warnings. Examples of such warning messages:\n\n{ws[0]}\n{ws[1]}\n{ws[2]}"
                    ] + warnings
            if warnings:
                state["warnings"] = warnings

        return state

    def _send_state(self, state: str = "PROGRESS", with_warnings: bool = False) -> None:
        """
        Update the broker result state of the current task
        Skips warnings unless otherwise specified (they can be many)

        :param state: The state of the task
        :param with_warnings: Whether to include warnings in the state
        """
        to_send = {"status": state, **self.get_state(with_warnings=with_warnings)}

        for backend in self._result_backends:
            backend.store_result(self._message, to_send, 60 * 60 * 24 * 1000)

    def info(self, *s, **kwargs):
        """
        Log an info message
        """
        text = "\n".join(str(k) for k in s)
        self._latest_infos.append(text)
        self._latest_infos = self._latest_infos[-10:]
        self._send_state(with_warnings=False)
        LoggerHelper.info(*s, **kwargs)

    def warning(
        self,
        *s,
        collapse: Optional[str] = None,
        exception: bool = False,
        send: bool = False,
    ) -> None:
        """
        Log a warning message

        :param s: The messages to log
        :param collapse: The type of warning (to group similar warnings)
        :param exception: Add the exception to the log
        :param send: Whether to send the state to the frontend right now
        """
        text = "\n".join(str(k) for k in s)

        if exception:
            text += "\n " + traceback.format_exc(limit=1)

        LoggerHelper.warning(*s, exception=exception)

        if collapse:
            self._grouped_warnings.setdefault(collapse, [])
            self._grouped_warnings[collapse].append(text)
        else:
            self._warnings.append(text)

        if send:
            self._send_state(with_warnings=True)

    def error(self, *s, exception: Exception = None) -> None:
        """
        Log an error message (and send the state to the frontend)

        :param s: The messages to log
        :param exception: Add the exception to the log
        """
        text = "\n".join(str(k) for k in s)

        if exception:
            text += f"\n\n{traceback.format_exc()}"

        LoggerHelper.error(*s, exception=exception)

        self._errors.append(text)
        self._send_state()

    def progress(
        self,
        current: int = 0,
        total: int = None,
        title: str = "",
        key: Optional[str] = None,
        end: bool = False,
        display: bool = False,
        send: bool = True,
        **kwargs,
    ) -> None:
        """
        Log the progress of a task

        :param current: The current progress
        :param total: The total progress
        :param title: The title of the task
        :param key: The key to identify the task
        :param end: Whether to terminate the progress tracking
        :param display: Whether to display the progress in the console
        :param send: Whether to send the state to the frontend (default: True)
        """
        if key is None:
            key = title

        if end:
            try:
                del self._progress[key]
            except KeyError:
                pass
            if send:
                self._send_state(with_warnings=False)
            return

        self._progress[key] = {
            "current": current,
            "total": total,
            "context": title,
        }

        if display:
            LoggerHelper.progress(current, total, title=title, **kwargs)

        if send:
            self._send_state(with_warnings=False)

    def iterate(
        self,
        iterable: Iterable[T],
        title: str = "",
        total: Optional[int] = None,
        rate_limit: float = 1.0,
    ) -> TqdmProgress:
        """
        Monitor the progress of iterating an iterable (through tqdm)

        :param iterable: The iterable to monitor
        :param title: The title of the task
        :param total: The total number of items in the iterable
        :param rate_limit: The minimum interval between updates
        """

        def progress_callback(prog: TqdmProgress):
            self.progress(prog.n, prog.total, title=title, key=prog.pos)

        def end_callback(prog: TqdmProgress):
            self.progress(end=True, key=prog.pos)

        return TqdmProgress(
            iterable,
            vc_progress_callback=progress_callback,
            vc_end_callback=end_callback,
            desc=title,
            mininterval=rate_limit,
            total=total,
        )


class LoggedResults(Results):
    """
    A class to store the results of a task in Dramatiq backend
    """

    def before_process_message(self, broker, message):
        # store a result saying Progress
        store_results, result_ttl = self._lookup_options(broker, message)
        if store_results:
            logger = JobLogger.getLogger(create=True)
            logger.register_backend(self.backend)


def console(msg, color="bold", e: Exception = None, log=True):
    msg = (
        f"\n\n\n\n[{get_time()}]\n{get_color(color)}{pprint(msg)}{ConsoleColors.end}\n"
    )
    if e:
        msg += f"\nStack Trace:\n{get_color('red')}{traceback.format_exc()}{ConsoleColors.end}\n"

    if log:
        base_logger.info(msg)
        return
    print(msg)


class LoggingTaskMixin:
    """
    A class mixin to log the progress of a task
    """

    def __init__(self, logger: TLogger, *args, **kwargs):
        self.jlogger = logger
        super().__init__(*args, **kwargs)

    def print_and_log(self, s, e: Exception = None, **kwargs) -> None:
        console(s, e=e, **kwargs)
        if e:
            self.jlogger.error(s, exception=e)
            return
        self.jlogger.info(s)

    def print_and_log_info(self, s) -> None:
        console(s)
        self.jlogger.info(s)

    def print_and_log_warning(self, s) -> None:
        console(s, color="yellow")
        self.jlogger.warning(s)

    def print_and_log_error(self, s, e: Exception = None) -> None:
        console(s, color="red", e=e)
        self.jlogger.error(s, exception=e)

    def run_task(self, *args, **kwargs):
        result = super().run_task(*args, **kwargs)
        return result


# def send_update(experiment_id, tracking_url, event, message):
#     # TODO delete
#     response = requests.post(
#         url=tracking_url,  # TODO delete
#         data={
#             "experiment_id": experiment_id,
#             "event": event,
#             "message": message or "",
#         },
#     )
#     response.raise_for_status()
#     return True
