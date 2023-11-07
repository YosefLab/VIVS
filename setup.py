#!/usr/bin/env python

import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name="vivs",
        version="0.1.1",
        packages=["vivs"],
        author="Pierre Boyeau",
        author_email="pierre.boyeau@gmail.com",
        description="""
            VIVS (Variational Inference for Variable Selection)
            is a Python packageto identify molecular
            dependencies in omics data.
        """,
    )
