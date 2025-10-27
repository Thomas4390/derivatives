import sys

def run_tests(*args):
    import argparse
    import unittest

    # To use the dorion_francois package, we need to add its parent directory to the sys.path
    sys.path.append('../..')

    # Importing the modules with a list_test_classes function
    import dorion_francois.garch as garch

    # Listing the modules with a list_test_classes function
    modules = {'garch' : garch}

    parser = argparse.ArgumentParser(description="Testing the dorion_francois package")

    default = list(modules.keys())
    parser.add_argument("modules", nargs='*', help="Modules to test (default: %s)"%default, default=default)

    parser.add_argument("--update", default=False, action="store_true", help="Update the expected results of the tests")

    configs = parser.parse_args(args)
    
    runner = unittest.TextTestRunner()
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for module in configs.modules:
        test_classes = modules[module].list_test_classes(configs)
        for test_class in test_classes:
            suite.addTests(loader.loadTestsFromTestCase(test_class))

    # Running the test suite
    runner.run(suite)

if __name__ == '__main__':
    args = sys.argv[1:]
    if False: # DBG, to be removed
        args = ['--update']
    run_tests(args)

#sananda: if __name__ == '__main__':
#sananda:     import os
#sananda:     os.system('python -m unittest -v dorion_francois.black_merton_scholes')
#sananda:     os.system('python -m unittest -v dorion_francois.volatility_models')
