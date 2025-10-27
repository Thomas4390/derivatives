from abc import ABC
import argparse
from datetime import datetime
import gc
import inspect
import logging
import multiprocessing
import os
from pathlib import Path
import pickle
import shutil
import socket
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from IPython.display import HTML, display

from dorion_francois.toolkit import is_jupyter_notebook

class CustomLogger(logging.Logger): 
    """
    Recommended usage is to:
      1. Create a logger with the name `__name__` in each module.
      2. Call the `set_current_task` method at the beginning of each task to set the log file name.

    The logger can have up to three active handlers:
      1. A StreamHandler that logs to the console. This handler is only active in the main process.
      2. A process-specific FileHandler that logs to a file. The log file name is the process name.
      3. A task-specific FileHandler that logs to a file. The log file name is the task ID.
    The first two handlers are always active, while the third handler is set by the `set_current_task` method.

    This class should not be instantiated directly, but through the :func:`get_logger` function.
    """

    # Shorthand, so that users don't have to `import logging`
    NOTSET = logging.NOTSET      
    DEBUG = logging.DEBUG        
    INFO = logging.INFO          
    WARNING = logging.WARNING    
    ERROR = logging.ERROR        
    CRITICAL = logging.CRITICAL

    # List of all existing loggers, used in set_current_task to update the handlers of all loggers at once
    existing_loggers: list["CustomLogger"] = []

    def __init__(self, name: str, level = logging.INFO): # name is not strictly necessary, but it is good practice to pass it
    
        super().__init__(name)
        CustomLogger.existing_loggers.append(self)
        self.current_task_handler: CustomFileHandler = None # The current task's file handler
        self.process_handler: CustomFileHandler = None # Process-specific log file. Defined later in set_current_task

        # Set the logging level
        self.setLevel(level)

        ### Set the logger handler
        process_name = multiprocessing.current_process().name
        if process_name == "MainProcess": # Main process logs to console
            self.console_handler = logging.StreamHandler()
            self.console_handler.setFormatter(
                logging.Formatter("%(levelname)s - %(message)s")
            )
            self.addHandler(self.console_handler)

            process_handler = self._get_file_handler(multiprocessing.current_process().name)
            self.addHandler(process_handler)


    def file_only_log(self, message, level=logging.INFO):
        """Log a message to the file handler only."""
        # Store current console handler level
        console_level = self.console_handler.level
        
        # Temporarily set console handler to a higher level
        self.console_handler.setLevel(logging.CRITICAL + 1)
        
        # Log the message (goes only to file since console level is too high)
        self.log(level, message)
        
        # Restore original console level
        self.console_handler.setLevel(console_level)

    def set_current_task(self, task_id: str):
        """Set the output log file to the task's ID."""
        task_handler = self._get_file_handler(task_id)
        process_handler = self._get_file_handler(multiprocessing.current_process().name)

        # Loop through all existing loggers and update their task handlers
        for logger in CustomLogger.existing_loggers:
            # Remove the old task handler and add the new one
            if logger.current_task_handler is not None:
                logger.removeHandler(logger.current_task_handler)
                logger.current_task_handler.close()

            logger.addHandler(task_handler)
            logger.current_task_handler = task_handler

            # Update the process handler to the new log folder
            # TODO: This should probably be in a separate method, as it only needs to be done once.
                    # We could use ProcessPoolExecutor's initializer feature to do so.
            if logger.process_handler is not None:
                logger.removeHandler(logger.process_handler)
                logger.process_handler.close()

            logger.addHandler(process_handler)
            logger.process_handler = process_handler

    def _get_file_handler(self, log_file_name):
        """
        Internal method to set a new file handler for the logger
        Args:
            log_file_name: The name of the log file
        """

        # Whether the log folder should be cleaned up on FileHandler creation (only at program start)
        should_cleanup = multiprocessing.current_process().name == "MainProcess" and self.name != "cleanup_error"
        handler = CustomFileHandler(str(log_file_name), should_cleanup=should_cleanup)

        return handler

    def _log(self, level: int, msg, *args, **kwargs) -> None:
        """Internal logging function called by all other logging functions.
        Overridden to adapt the message format to the current environment, depending on the message type."""

        # If the message is a numpy array or a pandas Series, convert it to a DataFrame
        if isinstance(msg, np.ndarray):
            msg = pd.DataFrame(msg)
        if isinstance(msg, pd.Series):
            msg = msg.to_frame().transpose()

        # If the message is a pandas DataFrame and we are in a Jupyter notebook, display it as HTML            
        if isinstance(msg, pd.DataFrame):
            if is_jupyter_notebook():
                self.log(level, "")
                display(HTML(msg.to_html()))
                return
            else:
                # Otherwise, display it as a string
                msg = "\n" + msg.to_string()

        # If the message is a Namespace object (configs), print its attributes one by one
        if isinstance(msg, argparse.Namespace):
            self._log(level, "configs:", *args, **kwargs)
            for key in vars(msg):
                self._log(
                    level, "  %s: %s" % (key, repr(getattr(msg, key))), *args, **kwargs
                )
            return
        
        # If the message is a dictionary, print its key-value pairs one by one
        if isinstance(msg, dict):
            self._log_dict(level, msg, *args, **kwargs)
            return
        
        ### Add the caller's name and line number to the message
        stack = inspect.stack()
        caller_frame = stack[2] # Get the caller's frame
        caller_fname = Path(caller_frame.filename).name # Get the filename without the path
        msg = f"{caller_fname}:{caller_frame.lineno} - {msg}"


        super()._log(level, msg, *args, **kwargs)
    

    def _log_dict(self, level: int, msg: dict, *args, **kwargs) -> None:
        """Log a dictionary. Each key-value pair is logged on a separate line.
        Multi-line strings are indented properly and logged line by line.
        """

        self._log(level, "dict:", *args, **kwargs)
        for key, value in msg.items():
            # If the value is a multi-line string, log it with proper indentation
            if isinstance(value, str) and "\n" in value:
                # Format with proper indentation
                indented_str = value.replace("\n", "\n    ")  # Add indentation to each line
                
                # Log the header first
                self._log(level, f"  {key}:", *args, **kwargs)
                
                # Log each line of the diff separately
                for line in indented_str.split("\n"):
                    self._log(level, f"    {line}", *args, **kwargs)
            else:
                self._log(
                    level, "  %s: %s" % (key, repr(value)), *args, **kwargs
                )

class _LazyLogger:
    """Acts as a wrapper, instantiating a logger when it is used, not merely declared."""
    
    __warning_handler_set = False
    """All instances share this flag, set True once when the first logger actual is instantiated and the warning handler is set."""

    @classmethod
    def set_warning_handler(cls):
        """Setting the warning handler upon first call."""
        if not cls.__warning_handler_set:
            warnings.showwarning = custom_warning_handler
            cls.__warning_handler_set = True

    def __init__(self, *args, **kwargs):
        """Cache the arguments provided the user, to later instantiate the logger when it is used."""
        self.__logger_args = (args, kwargs)
        self.__logger = None
        #self.__lock = threading.Lock()  # Ensure thread safety

    def __getattr__(self, name):
        """Create the actual logger on first use and forward it's attribute through the lazy logger."""
        # Avoid initializing for private or dunder attributes
        if name.startswith('__'):
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        
        if self.__logger is None:
        #with self.__lock:  # Ensure only one thread initializes the logger
            if self.__logger is None:  # Double-checked locking
                args, kwargs = self.__logger_args
                self.__logger = CustomLogger(*args, **kwargs)

                # Setting the warning handler upon the first logging avoid potential issues if warnings are issued by dependencies upon package creation.
                self.set_warning_handler()

        return getattr(self.__logger, name)
# #@Sofiane / Xavier: strangely, setting the warning handler late did not work properly... Comment out the following # line and see why warnings make their way to the shell.
# _LazyLogger.set_warning_handler()
## DOES NOT appear to be the problem; warning still make it to the shell after adding the line above!

class classproperty:
    def __init__(self, fget):
        self.fget = fget
    def __get__(self, obj, cls):
        return self.fget(cls)
    
class CustomFileHandler(logging.FileHandler):
    """Custom file handler that outputs to a file in the logs folder.
    Creates a new log folder for each run, and keeps a specified number of historical log folders.
    """
    backup_count = 10 # Number of historical log folders to keep
    log_folder = Path("./logs") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #TODO: Instead of _LazyLogger: The base of log_folder should be a `@classproperty def log_folder_base:` which 
    # * Is loaded from .tasks_cfh_logs if it exists, otherwise
    # * Is loaded from .tasks_cfh_logs_default if it exists, otherwise
    # # Defaults to ./logs
    # This would allow DV to define its own default in .tasks_cfh_logs_default, and overwrite it on clusters by sending .tasks_cfh_logs along with the code. This would probably be more robust than _LazyLogger

    #BUGGED!! with memray.Tracker(CustomFileHandler.memray_folder / f"{os.getpid()}.bin"):
    # memray_folder: Path = Path("./data/cache/memray") # Not written to by CustomLogger, but still counts as a log folder

    def __init__(self, log_file_name: str, should_cleanup=False):
        """Create a new file handler with the specified log file name.
        Args:
            log_file_name: The name of the log file
            should_cleanup: Whether to clean up old log folders
        """

        # Sanitize the log file name
        log_file_name = log_file_name.replace(os.path.sep, "_")

        # Create a unique filename for the run
        log_file_path = self.log_folder / f"{log_file_name}.log"
        # Create the logs folder if it does not exist
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if should_cleanup:
            self._cleanup_old_logs()
        
        super().__init__(log_file_path, mode="a")
        self.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

    @classmethod
    def set_log_folder(cls, log_folder):
        """
        Sets the log folder using the provided timestamp. This ensures all processes share the same
        folder name for logs in parallel execution, which keeps logs organized.
        """
        cls.log_folder = log_folder

    def _cleanup_old_logs(self):
        # Count folders in the logs folder, if there are more than backup_count, remove the oldest
        folders = sorted(self.log_folder.parent.glob("*"), key=os.path.getmtime)        
        
        # Do not count or remove files, or folders that contain a file named __permanent__
        tmp_folder = lambda fldr: fldr.is_dir() and not (fldr / "__permanent__").exists()
        folders = [fldr for fldr in folders if tmp_folder(fldr)]

        # While there are at least backup_count files, remove the oldest
        while len(folders) > self.backup_count:
            oldest = folders.pop(0)
            try:
                # Remove the folder and all its contents
                shutil.rmtree(oldest)
            except Exception as e:
                get_logger("cleanup_error").error(f"Error while cleaning up logs: {e}")

def get_logger(name: str) -> CustomLogger:
    """Returns a logger with the specified name."""
    return CustomLogger(name)

#_LL: def get_logger(name: str) -> _LazyLogger:
#_LL:     """Returns a logger with the specified name."""
#_LL:     return _LazyLogger(name) # CustomLogger(name)

logger = get_logger(__name__)


def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    """Custom warning handler that logs the warning message to the logger."""
    logger.warning(f"{category.__name__}: {message}")
warnings.showwarning = custom_warning_handler

class TaskError(Exception):
    def __init__(self, task_id, message):
        super().__init__(message)
        self.task_id = task_id

    def message(self):
        return f"{self.__class__.__name__}: {str(self)}"

    def describe(self):
        return (self.task_id, self.message())

class Task: #(ABC):
    """Base class for handling tasks.
    
    This task should never be instantiated.

    NOTE: The report.py `report_score` and `_report_scores` highlight that it might be wise to move class variables herein to attributes of a TaskSet instance, allowing for nested task sets...
    """
    complete = []
    incomplete = {}

    task_management_errors = [] # Further identifying these errors
    error_class = TaskError
    specialized_task_class = None

    @staticmethod
    def build(*args, **kwargs):
        TaskClass = Task if Task.specialized_task_class is None else Task.specialized_task_class
        return TaskClass(*args, **kwargs)

    @staticmethod # classmethod is more general, less readable...
    def sort_task(result):
        if isinstance(result, IncompleteTask):
            class_and_more = result.description.split(':',1)
            cls = globals().get(class_and_more[0])
            if cls is not None and issubclass(cls, Task.error_class):
                Task.task_management_errors.append(result.task_id)
            Task.incomplete[result.task_id] = result.description
        else:
            Task.complete.append(result)
        return len(Task.complete), len(Task.incomplete)

    #RM? @staticmethod
    #RM? def task_id_and_args(args):
    #RM?     if isinstance(args, tuple):
    #RM?         return args[0], args[1:]
    #RM?     return args, ()

    def __init__(self, function, task_id, *args, **kwargs):
        """Create a Task instance, waiting for execution.
        
        Args:
            function (Callable):
                The function that will be called in execute.
            task_id:
                Any unique identifier, usually an `int` or a `str`.
            *args:
                Will be forwarded to the function to run by `execute`.
            **kwargs:
                host (str):
                    The name of the host machine.
                log_folder (str):
                    If provided, `execute` sets `CustomFileHandler.set_log_folder(log_folder)` before `logger.set_current_task(self.task_id)`
                **other kwargs are ignored.**
        """
        self.function = function
        self.task_id = task_id
        self.args = args

        self.host = kwargs.pop('host',socket.gethostname())
        self.log_folder = kwargs.pop("log_folder", None)

    def execute(self):
        """Executes a task, aligning the logger with the task ID and the run start time."""
        # log_folder is set by the parallelize function
        if self.log_folder is not None:
            CustomFileHandler.set_log_folder(self.log_folder)
        
        logger.set_current_task(self.task_id)
        logger.info(f"Starting task {self.function.__name__} for ID: {self.task_id}")
        
        result = self.function(self.task_id, *self.args)
        
        logger.info(f"Task {self.function.__name__} for ID: {self.task_id} completed")

        return result
    
    def finalize(self):
        """Handle any post-processing needed after task completion.

        Override this method in specialized task classes for custom cleanup.
        """
        pass

class IncompleteTask(Task):
    """Wrapper for an incomplete task's results."""
    def __init__(self, task_id, description):
        self.task_id = task_id
        self.description = description

#(ABC): class LocalTask(Task):
#(ABC):     """By default, the task is ran on the local host."""
#(ABC):     def __init__(self, task_id, host=socket.gethostname()):
#(ABC):         self.task_id = task_id
#(ABC):         self.host = host
#(ABC): 
#(ABC):     def execute(self, function, *args, **kwargs):
#(ABC):         """Executes a task, aligning the logger with the task ID and the run start time."""
#(ABC):         # log_folder is set by the parallelize function
#(ABC):         log_folder = kwargs.pop("log_folder", None)
#(ABC):         if log_folder is not None:
#(ABC):             CustomFileHandler.set_log_folder(log_folder)
#(ABC):         
#(ABC):         logger.set_current_task(self.task_id)
#(ABC):         logger.info(f"Starting task {function.__name__} for ID: {self.task_id}")
#(ABC):         result = function(self.task_id, *args)
#(ABC):         logger.info(f"Task {function.__name__} for ID: {self.task_id} completed")
#(ABC): 
#(ABC):         return result

def split_dataframe_by_group(df, group_col, n_parts):
    """Splits a dataframe in approximately equal parts, respecting integrity of groups.
    
    Args:
        df (pd.DataFrame):
            Must have a column `group_col`.
        group_col (str):
            The name of the column along which to group. The function ensure that all rows with the sample value of `df[group_col]` are in the same subsample.
        n_parts (int):
            The function attempts to partition `df` into `n_parts` equally-sized partitions, but prioritizing that all rows with the sample value of `df[group_col]` are in the same subsample.
    
    Returns:
        list:
            A list of dataframes that partition the `df` in elements of closest possible size, ensuring that all rows with the sample value of `df[group_col]` are in the same sub-dataframe.
            
    Example:
    ```
    df = pd.DataFrame({
        "group": ["A"] * 50 + ["B"] * 30 + ["C"] * 20 + ["D"] * 100,
        "data": range(200)
    })
    splits = split_dataframe_by_group(df, "group", 3)
    check = []
    for no, df in enumerate(splits):
        next = np.unique(df["group"])
        assert ~np.any( np.isin(next,check) )
        check.extend( list(next) )
        print(f"Worker {no} gets {len(df)} rows")    
    ```
    Worker 0 gets 100 rows
    Worker 1 gets 50 rows
    Worker 2 gets 50 rows    
    """
    groups = df.groupby(group_col)
    # get group sizes as a DataFrame
    group_sizes = groups.size().reset_index(name="size")
    # sort groups by size (largest first)
    group_sizes = group_sizes.sort_values("size", ascending=False)
    
    # initialize partitions and total sizes for each partition
    partitions = [[] for _ in range(n_parts)]
    part_sizes = [0] * n_parts
    
    # Assign each group to the partition with currently smallest total size
    for _, row in group_sizes.iterrows():
        group_value = row[group_col]
        size = row["size"]
        idx = np.argmin(part_sizes)
        partitions[idx].append(group_value)
        part_sizes[idx] += size
    
    # Create dataframe slices for each partition
    slices = [df[df[group_col].isin(part)].copy() for part in partitions]
    return slices

def failsafe(task, **kwargs):
    failsafe_plot = kwargs.pop('_failsafe_plot',False)
    if failsafe_plot:
        from dorion_francois.plot_utils import plt

    try:
        return task.execute()
    except KeyboardInterrupt as err:
        sys.exit(1) # raise err?
    except Exception as err:
        logger.error(f"An error occurred: {traceback.format_exc()}")
        return IncompleteTask(task.task_id, f"{err}\n{traceback.format_exc()}")
    
    # The finally block is executed regardless of whether an exception was raised or not, and placing return statement in the finally block would override any return value from the try block or the except block.
    finally:
        if failsafe_plot:
            plt.close()
        gc.collect()
            
def no_failsafe(task, **kwargs):
    return task.execute()

def _df_tasks_kwargs(**kwargs):
    no_failsafe_flag = kwargs.pop('no_failsafe',False)
    progress_bar = kwargs.pop('progress_bar',False)
    pool_size = kwargs.pop('pool_size',None)

    assert "log_folder" not in kwargs
    kwargs["log_folder"] = CustomFileHandler.log_folder

    return no_failsafe_flag, progress_bar, pool_size, kwargs

def parallelize(function, task_args, *args, **kwargs):
    """Calls the function in parallel for each element in task_args.

    Args:
        function (Callable): 
            Function to call in parallel
        task_args (list): 
            Where each element is a tuple of arguments to pass to the function. The first element of each tuple will always be considered as a task ID; ensure that it is distinct for all your tasks.
        *args (tuple): 
            Additional arguments that are common to all tasks. For each element `t_args` of `task_args`, `args` are added to `t_args` (i.e. `(*t_args,*args)`).
        **kwargs (dict):
            Is NOT provided to `function`, but used to configure the behavior of this function.
    """
    import atexit
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import dorion_francois.toolkit as dftk

    def cleanup(executor):
        executor.shutdown(wait=False)    

    # By default, do not let the exceptions interfere with the parallelization
    no_failsafe_flag, progress_bar, pool_size, kwargs = _df_tasks_kwargs(**kwargs)

    with ProcessPoolExecutor(max_workers=pool_size) as executor:
        atexit.register(cleanup, executor) # Register the cleanup function

        if no_failsafe_flag:
            submit = lambda task: executor.submit(no_failsafe, task, **kwargs)
        else:
            submit = lambda task: executor.submit(failsafe, task, **kwargs)

        # Allowing for just-in-time preprocessing is not trivial...
        max_workers = executor._max_workers
        task_iterator = iter(task_args)
        active_futures = [] 
        future_to_task = {}
        results = []

        def submit_next_task():
            """Pre-process and submit the next available task just-in-time"""
            try:
                t_args = next(task_iterator) #? Task.task_id_and_args( next(task_iterator) )
                
                # Task.specialized_class can implemented (almost) just-in-time pre-processing in their constructor 
                task = Task.build(function, t_args, *args, **kwargs)
                
                # Submit immediately after pre-processing
                future = submit(task)
                future_to_task[future] = task
                return future
                
            except StopIteration:
                return None
        
        # Fill initial window (avoid performing the preprocessing all at once)
        start_time = time.perf_counter()
        for _ in range( min(max_workers, len(task_args)) ):
            future = submit_next_task()  # This returns a future, not a task
            if future:
                active_futures.append(future)
        elapsed_time = time.perf_counter() - start_time
        logger.info(f"Initial window fill took {elapsed_time:.3f} seconds")

        with dftk.progress_bar(total=len(task_args), 
                                desc="Progress", unit="task", disable=not progress_bar) as pbar:
            while active_futures:
                # Wait for at least one future to complete
                done_future = next(as_completed(active_futures))
                
                # Get the result and associated task
                task = future_to_task[done_future]
                result = done_future.result()
                task.finalize()  # Call finalize after getting result
                
                # Update progress and collect results
                #   Now that task instances are created above, could (e.g.) task.incomplete be used instead of "parsing" the result?
                success_count, failure_count = Task.sort_task(result)
                pbar.set_postfix({'success': success_count, 'failure': failure_count})
                pbar.update(1)
                results.append(result)
                
                # Clean up completed future
                active_futures.remove(done_future)
                del future_to_task[done_future]
                
                # Submit next task to maintain the sliding window
                next_future = submit_next_task()
                if next_future:
                    active_futures.append(next_future)

        # Unregister the cleanup function when it is no longer needed
        atexit.unregister(cleanup)
    return results

def run(function, tasks_args, *args, **kwargs):
    """Runs `function` for each element of `tasks_args`, forwarding `*args` to function.

    kwargs controls the behavior of this function:    
        single_thread (bool, default False):
            If True, runs `function` in a for loop. Otherwise, runs it on a ProcessPoolExecutor.
        
        no_failsafe (bool, defaults to `single_thread`)
            If True, `function` is not wrapped in the `failsafe` function defined in this module. The default behavior is to use `failsafe` in parallel and not in the for loop.

    Other kwargs are forwarded to parallelize when `single_thread` is False.
    """
    import dorion_francois.toolkit as dftk
    
    single_thread = kwargs.pop('single_thread',False)
    if single_thread:
        results = []        
        no_failsafe_flag, progress_bar, pool_size, kwargs = _df_tasks_kwargs(**kwargs)
        with dftk.progress_bar(total=len(tasks_args), desc="Progress", unit="task") as pbar:
            for t_args in tasks_args:
                t_args = t_args if isinstance(t_args,tuple) else (t_args,)
                task = Task.build(function, *t_args, *args, **kwargs)
                if no_failsafe:
                    res = task.execute()
                else:
                    res = failsafe(task, **kwargs)
                Task.sort_task(res)
                results.append(res)
                pbar.update(1)                
        return results

    return parallelize(function, tasks_args, *args, **kwargs)

AD_HOC_N_WORKERS = """
import psutil
n_physical_cores = psutil.cpu_count(logical=False)
n_logical_cores = psutil.cpu_count(logical=True)
n_workers = int((n_physical_cores-1) * n_logical_cores / n_physical_cores)
print(f'n_workers = {n_workers} ({n_physical_cores} / {n_logical_cores})')
"""