"""Various utility functions that did not find their way into a more "specialized" package.

Many such functions are also in jupyter_notebook. However, jupyter_notebook was really intended 
to "initialize" notebooks in a homogeneous manner, and utility functions found there way in there 
not so much by design, more by laziness. A careful migration will come... progressively.

A first "category" of helper functions appear to be related to date management. These might
potentially be moved to a "calendar_tools.py" package in the future.
"""
import random
import time
import warnings

import inflect
import numpy as np
import pandas as pd
import pandas.io.formats.style
from IPython.display import HTML

inflect_engine = inflect.engine()
plural = inflect_engine.plural

class WarningCounter(warnings.catch_warnings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warning_count = 0

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, *exc_info):
        self.warning_count = len(self.log)
        super().__exit__(*exc_info)

class struct(dict):
    """Matlab-inspired placeholder built on a dictionary.

    An instance of this class allows accessing/setting elements of the dictionary just like we would for 
    attributes. Syntactically, this can be very convenient, especially given that pd.DataFrame also 
    allows accessing columns as attributes.
    """
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        else:
            raise AttributeError("No such attribute: " + attr)
        # return super().__getattr__(attr)

    def __setattr__(self, attr, value):        
        self[attr] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)        

def assert_unique(values):
    """Asserts that np.unique returns a sole value, and return it.""" 
    val = np.unique(values)
    assert len(val)==1, 'Expecting a unique value'
    return val[0]

def excel_sheet_to_df(filename, sheet_name, columns, index_col=0):
    """Reads an Excel file and returns the specified sheet as a DataFrame.
    
    If the sheet does not exist, an empty DataFrame is returned with the specified columns.
    """
    try:
        df = pd.read_excel(filename, sheet_name=sheet_name, index_col=index_col)
    except ValueError as err:
        assert str(err) == f"Worksheet named '{sheet_name}' not found"
        df = pd.DataFrame(columns=columns)
    return df

def unique_or_none(values, assert_unique=True):
    """Returns the sole np.unique value in `values`, or None if empty.

    If np.unique returns more than one element, the default behavior is to fail an assertion test. 
    If `assert_unique` is False, then the function simply returns None.
    """
    if values.size==0:
        return None
    
    values = np.unique(values)
    if assert_unique:
        assert values.size==1, \
            "np.unique returns more than one element, the default behavior is to fail an assertion test."
        return values[0]
    return (values[0] if values.size==1 else None)
numunique = unique_or_none

def compile_to_shared_object(abs_path_to_c_file):
    import os
    import platform
    import sys

    assert abs_path_to_c_file.endswith('.c'), 'Expecting a .c file'
    c_file = abs_path_to_c_file
    so_file = abs_path_to_c_file[:-2] + '.so'
    the_folder, the_file = os.path.split(abs_path_to_c_file[:-2])
    platform_file = os.path.join(the_folder, '.'+the_file+'.platform')

    # Check if the .so file exists and if the .c file is newer
    if not os.path.exists(so_file) or os.path.getmtime(c_file) > os.path.getmtime(so_file):
        compile = True
    else:
        # Check if the .so file was compiled on a different platform
        if os.path.exists(platform_file):
            with open(platform_file, 'r') as f:
                compiled_platform = f.read().strip()
            if compiled_platform != platform.platform():
                compile = True
            else:
                compile = False
        else:
            compile = True

    if compile:
        os.system(f'gcc -shared -o {so_file} {c_file} -fPIC -lm')
        with open(platform_file, 'w') as f:
            f.write(platform.platform())
    return so_file

def is_jupyter_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type, e.g. ipython within PyCharm
    except NameError:
        return False      # Probably standard Python interpreter

def first_non_increasing_index(arr):
    """Return the index of the first non increasing value in an array
    
    If the array is strictly increasing, return None.
    """
    # Calculate the difference between consecutive elements
    differences = np.diff(arr)

    # Find the index of the first non-positive difference
    non_increasing_index = np.where(differences <= 0)[0]

    # Check if we found any non-increasing differences
    if non_increasing_index.size > 0:
        return non_increasing_index[0]
    return None

def find_the_code(func):
    import inspect

    # Get the source file and line number
    source_file = inspect.getsourcefile(func)
    line_number = inspect.getsourcelines(func)[1]

    print(f'Defined in file: {source_file}, line: {line_number}')    

def nan_equal_dataframe(df1, df2):
    equal = False
    shape_equal = df1.shape == df2.shape
    columns_equal = df1.columns.equals(df2.columns)            
    if shape_equal and columns_equal:
        eq_entries = (df1 == df2)
        eq_nan = df1.isna() & df2.isna()
        eq_nan_obj = (df1.isna() & (df2=='')) | ((df1=='') & df2.isna())
        equal = np.all(eq_entries | eq_nan | eq_nan_obj)
    return equal

def nancorr(ar, a2=None):
    if a2 is not None:
        return nancorr([ar, a2])
    
    assert isinstance(ar, list), "Other classes than list not implemented yet"
    
    import numpy.ma as ma
    msk = np.ones(ar[0].shape, dtype=bool)
    for no,a_i in enumerate(ar):
        a_i = ma.masked_invalid(a_i)
        msk = msk & ~a_i.mask
        ar[no] = a_i
        
    return np.corrcoef([nn[msk] for nn in ar])

def object_vars_to_string(obj):
    string = obj.__class__.__name__ + '(\n'    
    fields = vars(obj)
    for no,name in enumerate(fields):
        if name.startswith('_'):
            continue
        string += "    %s = %r" % (name, fields[name])
        if no < len(fields)-1:
            string += ','
        string += '\n'
    string += ')'
    return string    

def print_versions():
    """Versions of the critical dependencies"""
    from platform import python_version

    import scipy
    print('Python:',python_version())
    print('Numpy:',np.__version__)
    print('Pandas:',pd.__version__)    
    print('Scipy:',scipy.__version__)    

def printdf(df, T=True, head_tail=None):
    #import pdb; pdb.set_trace()
    
    if isinstance(df, type(np.array([]))):
        df = pd.DataFrame(df)
    
    if isinstance(df, pd.Series):
        if T:
            df = df.to_frame().transpose()
        else:
            df = df.to_frame()

    if head_tail is not None:
        from itertools import chain
        lx = df.shape[0]
        ix = chain(range(head_tail), range(lx-head_tail,lx))
        df = df.iloc[ix]

    #import pdb; pdb.set_trace()
    if not is_jupyter_notebook():
        print(df)
        return
            
    if isinstance(df, pd.io.formats.style.Styler):
        display(df)        
    else:
        display(HTML(df.to_html()))

def progress_bar(*args, **kwargs):
    _disabled = kwargs.pop('_disabled', False)
    if _disabled:
        from contextlib import contextmanager        
        @contextmanager
        def dummy_context_manager():
            class DummyPbar:
                def update(self, n=1):
                    pass
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass
            yield DummyPbar()
        return dummy_context_manager()

    if is_jupyter_notebook():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    return tqdm(*args, **kwargs)

__tic = [] # Last in, first out (LIFO)
def tic():
    """Marks the beginning of a time interval"""    
    global __tic
    __tic.append(time.perf_counter())

def toc(do_print=True):
    """Prints the time difference since the last tic (that was not toc'ed yet; LIFO)."""
    global __tic
    dt = time.perf_counter() - __tic.pop()
    if do_print:
        print("Elapsed time: %f seconds.\n"%dt)
    return dt

def touch(path):
    """Mimics the behavior of the 'touch' shell command.  

    From https://stackoverflow.com/a/12654798/21359768
    """
    import os
    with open(path, 'a'):
        os.utime(path, None)


#### Notebook utils:

def reload_local_modules():
    """Reload all modules in the current directory and its subdirectories.

    TEST BEFORE USING! This function is experimental and may not work as expected.
    
    # Dictionary to map variable names to module names
    loaded_modules = {}
    
    # Save the original __import__ function
    original_import = builtins.__import__
    
    def custom_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = original_import(name, globals, locals, fromlist, level)
        if name in loaded_modules:            
            loaded_modules[name].extend( list(fromlist) )
        else:
            loaded_modules[name] = list(fromlist)
        return module
    
    # Override the built-in __import__ function
    builtins.__import__ = custom_import    
    """
    import importlib
    import os
    import sys
    global loaded_modules

    current_dir = os.getcwd()
    for root, dirs, files in os.walk(current_dir):
        for file in files:
            if file.endswith(".py"):
                module_name = os.path.splitext(os.path.relpath(os.path.join(root, file), current_dir))[0].replace(os.sep, '.')
                if module_name in loaded_modules:
                    print(f"Reloading module: {module_name}")
                    importlib.reload(sys.modules[module_name])

                    # Update the global namespace with the previously loaded attributes
                    module = sys.modules[module_name]
                    for attr_name in loaded_modules[module_name]:
                        globals()[attr_name] = getattr(module, attr_name)

def show_hide_cell(for_next=False):
    """Return a link allowing to show/hide a cell.
    
    From https://stackoverflow.com/a/52664156/21359768
    """
    this_cell = """$('div.cell.code_cell.rendered.selected')"""
    next_cell = this_cell + '.next()'

    toggle_text = 'Show/hide cell'  # text shown on toggle link
    target_cell = this_cell  # target cell to control with toggle
    js_hide_current = ''  # bit of JS to permanently hide code in current cell (only when toggling next cell)
    #TODO: js_hide_current NOT WORKING! FIX IT
    
    if for_next:
        target_cell = next_cell
        toggle_text = 'Show/hide next cell'
        js_hide_current = this_cell + '.find("div.input").hide();'
        
        # To do: hide the current cell at the same time?
        # show_hide_cell()

    js_f_name = 'code_toggle_{}'.format(str(random.randint(1,2**64)))

    html = """
        <script>
            function {f_name}() {{
                {cell_selector}.find('div.input').toggle();
            }}

            {js_hide_current}
        </script>

        <a href="javascript:{f_name}()">{toggle_text}</a>
    """.format(
        f_name=js_f_name,
        cell_selector=target_cell,
        js_hide_current=js_hide_current, 
        toggle_text=toggle_text
    )
    return HTML(html)              

#### Date utils:

def subcalendar(calendar, after=None, first_date=None, last_date=None, before=None):
    """Subsample the given calendar, or DataFrame based on its index.    

    If the first argument is a `pd.DataFrame`, its index will be assumed to be the calendar to subsample 

    Args:
        calendar: 
            the calendar or `pd.DataFrame` to subsample from
        after: 
            excluding dates before or on the `after` date (mutually exclusive with `first_date`)
        first_date: 
            starting on first_date, included if it exists (mutually exclusive with `after`)
        last_date: 
            ending on last_date, included if it exists (mutually exclusive with `before`)
        before: 
            excluding dates after or on the `before` date (mutually exclusive with `last_date`)
    """
    df = None
    if isinstance(calendar, pd.DataFrame):
        df = calendar
        calendar = df.index
    if len(calendar)==0:        
        return (calendar if df is None else df)
    
    if isinstance(calendar[0],pd.Timestamp):
        timestamp = lambda date: date if not isinstance(date,str) \
                                        else pd.Timestamp(date+' 00:00:00')
    else:
        timestamp = lambda date: date if not isinstance(date,str) \
                                        else pd.Timestamp(date+' 00:00:00').date()    

    if after is not None:
        calendar = calendar[calendar > timestamp(after)]
        assert first_date is None, 'after and last_date cannot be used jointly'
    elif first_date is not None:
        calendar = calendar[calendar >= timestamp(first_date)]
        
    if last_date is not None:
        calendar = calendar[calendar <= timestamp(last_date)]
        assert before is None, 'first_date and before cannot be used jointly'            
    elif before is not None:
        calendar = calendar[calendar < timestamp(before)]

    if df is None:
        return calendar
    return df.loc[calendar]

def subsample(df, sx, pad=1):
    """Returns rows where dx is True, with the previous/next `pad` rows."""
    loc = np.array([df.index.get_loc(ix) for ix in df.index[sx]])
    xloc = loc
    for offset in range(-pad,pad+1):
        xloc = np.concatenate((xloc, loc+offset))
    xloc = np.maximum(0, xloc)
    xloc = np.minimum(xloc, len(df.index)-1)
    return df.iloc[np.unique(xloc)].sort_index()

def datetime2yyyymmdd(dt):
    if not hasattr(dt,'year'):
        dt = pd.to_datetime(dt)
    return int(1e4*dt.year + 1e2*dt.month + dt.day)

def yyyymmdd2timestamp(yyyymmdd):
    #breakpoint()
    dd     = np.mod(yyyymmdd, 100)
    yyyymm = (yyyymmdd - dd)/100;  
    mm     = np.mod(yyyymm, 100);
    yyyy   = (yyyymm - mm)/100;
    if not hasattr(yyyymmdd,'__len__'):
        return pd.Timestamp('%d-%02d-%02d'%(yyyy,mm,dd))
    return [pd.Timestamp('%d-%02d-%02d'%(yyyy[no],mm[no],dd[no])) for no in range(len(yyyymmdd))]

def date2str(dt):
    if isinstance(dt, np.datetime64):
        dt = pd.to_datetime(dt)
    return dt.strftime('%Y/%m/%d')
