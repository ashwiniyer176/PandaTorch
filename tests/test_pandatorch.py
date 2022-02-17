from pandatorch import __version__
import os


def test_version():
    assert __version__ == '1.0.0'


def test_cwd_has_rst():
    cwd = os.getcwd()
    assert os.path.exists(os.path.join(cwd, "README.rst")) == True
