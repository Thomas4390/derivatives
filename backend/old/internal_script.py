"""Allow absolute imports in internal scripts

From https://stackoverflow.com/questions/16981921/relative-imports-in-python-3

---
From: Aya
answered Jun 7, 2013 at 13:14
edited Mar 31 at 12:13 by jjv-liu
-
It's quite common to have a layout like this...

main.py
mypackage/
    __init__.py
    mymodule.py
    myothermodule.py
...with a mymodule.py like this...

#!/usr/bin/env python3

# Exported function
def as_int(a):
    return int(a)

# Test function for module  
def _test():
    assert as_int('1') == 1

if __name__ == '__main__':
    _test()
...a myothermodule.py like this...

#!/usr/bin/env python3

from .mymodule import as_int

# Exported function
def add(a, b):
    return as_int(a) + as_int(b)

# Test function for module  
def _test():
    assert add('1', '1') == 2

if __name__ == '__main__':
    _test()
...and a main.py like this...

#!/usr/bin/env python3

from mypackage.myothermodule import add

def main():
    print(add('1', '1'))

if __name__ == '__main__':
    main()
...which works fine when you run main.py or mypackage/mymodule.py, but fails with mypackage/myothermodule.py, due to the relative import...

from .mymodule import as_int
The way you're supposed to run it is by using the -m option and giving the path in the Python module system (rather than in the filesystem)...

python3 -m mypackage.myothermodule
...but it's somewhat verbose, and doesn't mix well with a shebang line like #!/usr/bin/env python3.

An alternative is to avoid using relative imports, and just use...

from mypackage.mymodule import as_int
Either way, you'll need to run from the parent of mypackage, or add that directory to PYTHONPATH (either one will ensure that mypackage is in the sys.path module search path). Or, if you want it to work "out of the box", you can frob the PYTHONPATH in code first with this...

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from mypackage.mymodule import as_int
It's kind of a pain, but there's a clue as to why in an email written by a certain Guido van Rossum...

I'm -1 on this and on any other proposed twiddlings of the __main__ machinery. The only use case seems to be running scripts that happen to be living inside a module's directory, which I've always seen as an antipattern. To make me change my mind you'd have to convince me that it isn't.

Whether running scripts inside a package is an antipattern or not is subjective, but personally I find it really useful in a package I have which contains some custom wxPython widgets, so I can run the script for any of the source files to display a wx.Frame containing only that widget for testing purposes.

---
From: vaultah
edited Dec 31, 2017 at 11:40
answered Jan 26, 2015 at 16:54
-
Solution #4: Use absolute imports and some boilerplate code
Frankly, the installation is not necessary - you could add some boilerplate code to your script to make absolute imports work.

I'm going to borrow files from Solution #1 and change standalone.py:

Add the parent directory of package to sys.path before attempting to import anything from package using absolute imports:

import sys
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError: # Already removed
    pass
Replace the relative import by the absolute import:

from package import module  # absolute import
standalone.py runs without problems:

vaultah@base:~$ python3 -i package/standalone.py
Running /home/vaultah/package/standalone.py
Importing /home/vaultah/package/__init__.py
Importing /home/vaultah/package/module.py
>>> module
<module 'package.module' from '/home/vaultah/package/module.py'>
>>> import sys
>>> sys.modules['package']
<module 'package' from '/home/vaultah/package/__init__.py'>
>>> sys.modules['package.module']
<module 'package.module' from '/home/vaultah/package/module.py'>
I feel that I should warn you: try not to do this, especially if your project has a complex structure.

As a side note, PEP 8 recommends the use of absolute imports, but states that in some scenarios explicit relative imports are acceptable:

Absolute imports are recommended, as they are usually more readable and tend to be better behaved (or at least give better error messages). [...] However, explicit relative imports are an acceptable alternative to absolute imports, especially when dealing with complex package layouts where using absolute imports would be unnecessarily verbose.
"""
import os
import sys
package = os.path.dirname(__file__) 
root,df = os.path.split(package)
assert df=='dorion_francois', "internal_script cannot be used if the root is package is not installed in a directory called dorion_francois"

# The root directory to sys.path
sys.path.append(root)


try:
    # Additionally remove the current file's directory from sys.path
    sys.path.remove(package)

    # A priori, this file is used in projects that shouldn't refer to my dev version of the package
    sys.path.remove('/Users/christian/Documents/python')
except ValueError: # Already removed
    pass
