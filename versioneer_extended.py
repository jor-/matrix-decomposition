import versioneer


def get_version():
    """Get the short version string for this project."""
    return versioneer.get_version().replace('.dev0+', '+dirty.').replace('.post', '.post0.dev')


def get_cmdclass():
    """Get the custom setuptools/distutils subclasses used by Versioneer."""
    return versioneer.get_cmdclass()
