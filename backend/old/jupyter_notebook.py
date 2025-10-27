"""Setup and tools for Jupyter notebooks"""
import os
import pdb
import random
import re
import sys
import time
import warnings

import inspect
from pprint import pprint

import numpy as np
import pandas as pd
import pandas.io.formats.style
pandas.options.display.max_columns = None
pandas.options.display.float_format = '{:,.2f}'.format
from IPython.display import clear_output, HTML
from IPython.display import Markdown as md

from .plot_utils import mpl, plt, mtick, mdates, gridspec, set_plt_defaults

## GENERAL HELPER FUNCTIONS ##
from .toolkit import printdf, print_versions, tic, toc, numunique

dollars = lambda amount: "{:,.2f}$".format(amount)

def nansum(array):
    array=array[np.isnan(array)==False]
    return np.sum(array)

def nanmean(array):
    array=array[np.isnan(array)==False]
    # there was a mistake here
    return np.mean(array)

def shift(arr, num, fill_value=np.nan):
    '''Weirdly, this is faster than np.roll, and less error prone

    shitf5 in gzc's Mar 7, 2017 at 7:12 answer
    https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array/42642326#42642326?newreg=3c33081e4689466189394f4f25ff78eb
    '''
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def yeardelta(timedelta):
    "On average, a year is approximately 365.2425 days when accounting for leap years."
    return timedelta/pd.Timedelta(365.2425, unit='D')

def pdb_on_warning(one_line_code, *args):
    import pdb
    warnings.simplefilter('error', *args)  # treat these warnings as exceptions
    try:
        res = eval(one_line_code)
    except:
        pdb.post_mortem(sys.exc_info()[-1])
    warnings.resetwarnings()
    return res

def update_progress(progress, txt='Progress'):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait = True)
    text = txt + ": [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)    

def split_duplicates(data):
    dup = data.index.duplicated(keep='first')
    if dup.sum() == 0:
        return data, pd.DataFrame([], index=data.index, columns=data.columns)
    return data[~dup], data[dup]

def multiple_replace(string, rep_dict):
    pattern = re.compile("|".join([re.escape(k) for k in sorted(rep_dict,key=len,reverse=True)]), flags=re.DOTALL)
    return pattern.sub(lambda x: rep_dict[x.group(0)], string)

#toolkit.py: __exceptions = {'day':'days'}
#toolkit.py: def plural(word, count=2):
#toolkit.py:     """https://www.geeksforgeeks.org/python-program-to-convert-singular-to-plural/"""
#toolkit.py:     if not (np.abs(count) > 1):
#toolkit.py:         return word
#toolkit.py:     
#toolkit.py:     if word in __exceptions:
#toolkit.py:         return __exceptions[word]
#toolkit.py:     
#toolkit.py:     # Check if word is ending with s,x,z or is
#toolkit.py:     # ending with ah, eh, ih, oh,uh,dh,gh,kh,ph,rh,th
#toolkit.py:     if re.search('[sxz]$', word) or re.search('[^aeioudgkprt]h$', word):
#toolkit.py:         return re.sub('$', 'es', word)
#toolkit.py:     
#toolkit.py:     # Check if word is ending with ay,ey,iy,oy,uy
#toolkit.py:     if re.search('[aeiou]y$', word):
#toolkit.py:         # Make it plural by removing y from end adding ies to end
#toolkit.py:         return re.sub('y$', 'ies', word)
#toolkit.py:  
#toolkit.py:     # Make the plural of word by adding s in end
#toolkit.py:     return word + 's'

__single_warning = []
def single_warning(w_msg):
    if w_msg not in __single_warning:
        warnings.warn(w_msg)
        __single_warning.append(w_msg)
    pass

def deprecated_function(old,new):
    msg = 'function %s is deprecated, use %s instead'%(old,new)
    def func(*args, **kwargs):
        raise RuntimeError(msg)

        
## Moved to toolkit.show_hide_cell
# def hide_toggle(for_next=False):
#     this_cell = """$('div.cell.code_cell.rendered.selected')"""
#     next_cell = this_cell + '.next()'
# 
#     toggle_text = 'Toggle show/hide'  # text shown on toggle link
#     target_cell = this_cell  # target cell to control with toggle
#     js_hide_current = ''  # bit of JS to permanently hide code in current cell (only when toggling next cell)
# 
#     if for_next:
#         target_cell = next_cell
#         toggle_text += ' next cell'
#         js_hide_current = this_cell + '.find("div.input").hide();'
# 
#     js_f_name = 'code_toggle_{}'.format(str(random.randint(1,2**64)))
# 
#     html = """
#         <script>
#             function {f_name}() {{
#                 {cell_selector}.find('div.input').toggle();
#             }}
# 
#             {js_hide_current}
#         </script>
# 
#         <a href="javascript:{f_name}()">{toggle_text}</a>
#     """.format(
#         f_name=js_f_name,
#         cell_selector=target_cell,
#         js_hide_current=js_hide_current, 
#         toggle_text=toggle_text
#     )
# 
#     return HTML(html)        

class struct_OLD: # Deprecated implementation. See dorion_francois.toolkit
    '''Matlab-inspired placeholder
    
    Using dictionaries would be more Pythonesque but makes the code a tad heavy at times. When
    efficiency is not a concern, we may use instances of this class.
    '''    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self,key)
            
    def __to_string__(self, printable):
        string = self.__class__.__name__ + '(\n'
        #breakpoint()
        is_field = lambda name: hasattr(self, name) and not inspect.ismethod(getattr(self, name))
        fields = self.__dict__
        for no,name in enumerate(fields):
            if printable(name) and is_field(name):
                string += "    %s = %r" % (name, getattr(self, name))
                if no < len(fields)-1:
                    string += ','
                string += '\n'
        string += ')'
        return string    
    
    def __str__(self):
        is_public = lambda name: not (name.startswith('__') and name.endswith('__')) \
                                    and not name.startswith('_'+self.__class__.__name__)
        return self.__to_string__(is_public)

    def __repr__(self):
        is_public = lambda name: not (name.startswith('__') and name.endswith('__')) \
                                    and not name.startswith('_'+self.__class__.__name__)
        return self.__to_string__(is_public).replace('\n','').replace(' ','')
    
