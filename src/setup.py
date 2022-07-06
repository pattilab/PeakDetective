from setuptools import setup, find_packages

setup(
    name='PeakDetective',  # How you named your package folder (MyLib)
    packages=["PeakDetective"],  # Chose the same as "name"
    version='0.0.4',  # Start with a small number and increase it with every change you make
    license='MIT',  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description='Curates and detects LC/MS peaks in metabolomics datasets',  # Give a short description about your library
    author='Ethan Stancliffe',  # Type in your name
    author_email='estancl1234@gmail.com',  # Type in your E-Mail
    url='https://github.com/e-stan/PeakDetective/',  # Provide either the link to your github or to your website
    download_url='https://github.com/e-stan/PeakDetective/archive/v0.0.4.tar.gz',  # I explain this later on
    keywords=['Metabolomics', 'LC/MS', "Deep Learning","semi-supervised learning","machine learning"],  # Keywords that define your package best
    install_requires=[  # I get to this in a second
        'pyteomics',
        'numpy',
        'scipy',
        "matplotlib",
        "tensorflow",
        "scikit-learn",
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',  # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',  # Again, pick a license
        'Programming Language :: Python :: 3.7',  # Specify which pyhton versions that you want to support
    ],
)