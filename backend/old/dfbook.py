"""Handles paths to chapters and saving figures.

Usually imported as follows: `import dorion_francois.dfbook as df`

By default, the path to a given chapter (e.g. chapter01) is "./chapter01". This behavior can be changed by
creating a single-line hidden file .dfbook.git containing the path to the book's chapters. 

Calling path.book_chapter(chapter) uses the `chapter` key (e.g. 'intro') to get the path to which the key 
maps (e.g. `os.path.join(__dfbook_git, 'chapter01')`) in the current structure of the book. One can call
`df.path.get_chapter_mappings()` to list the current mappings.

The `df.path.book_chapter` method can be made to create an empty directory when the target does not exist 
by setting 

```
df.path.make_chapter_directories = True # defaults to False
```

This modulates the behavior of the `df.savefig(chapter, fname)` function. If `df.path.book_chapter(chapter)` 
exists, the figure is saved; otherwise, the functions silently does nothing. If df.path.make_chapter_directories
is True, then the figure is always saved.
"""
import os
import matplotlib.pyplot as plt

__dfbook_git = '.'
__dfbook_file = '.dfbook_git'
if os.path.exists(__dfbook_file):
    with open(__dfbook_file, 'r') as fh:
        __dfbook_git = fh.read().rstrip()

__chapters = {'intro'                : os.path.join(__dfbook_git, 'chapter01'),
              'diffusion'            : os.path.join(__dfbook_git, 'chapter02'),
              'bms'                  : os.path.join(__dfbook_git, 'chapter03'),
              'discrete_time'        : os.path.join(__dfbook_git, 'chapter04'),
              'exotic_options'       : os.path.join(__dfbook_git, 'chapter05'),
              'numerical_methods'    : os.path.join(__dfbook_git, 'chapter06'),
              'teachings_from_smile' : os.path.join(__dfbook_git, 'chapter07_new'),
              'modeling_volatility'  : os.path.join(__dfbook_git, 'chapter07'),
              'jump_diffusion'       : os.path.join(__dfbook_git, 'chapter08')
              }

class __dfbook_path:
    def __init__(self):
        self.make_chapter_directories = False
        
    def get_chapter_mappings(self):
        return globals()['__chapters']
        
    def book_chapter(self, chapter):
        __chapters = self.get_chapter_mappings()
        book_chapter = __chapters[chapter]
        if not os.path.isdir(book_chapter) and self.make_chapter_directories:
            os.mkdir(book_chapter)
        return book_chapter

path = __dfbook_path()

def savefig(chapter, fname):
    """Will save the figure only if the chapter folder exists."""
    dirname = path.book_chapter(chapter)
    if os.path.exists(dirname):
        filename = os.path.join(dirname, fname)
        plt.savefig( filename )
    plt.show() # Ensure the figure is visible