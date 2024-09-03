from distutils.core import setup, Extension

ext_modules = Extension("hello_world", libraries=['hello_world_c'])

def main():
    setup(name="hello_world",
          version="1.0.0",
          description="Python interface for the hello_world C library function",
          author="Dr Oliver Sheridan-Methven",
          author_email="oliver.sheridan-methven@hotmail.co.uk",
          # packages=['hello_world'],
          # package_data={'hello_world': ['hello_world_c.dylib']}
          ext_modules=ext_modules,
        # ext_modules=[Extension("hello_world", ["hello_world.c"])])
          )

if __name__ == "__main__":
    main()