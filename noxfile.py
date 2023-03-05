import nox


@nox.session(python=["3.8", "3.9", "3.10", "3.11"])
def tests(session):
    session.install("-r", "requirements_dev.txt")
    session.run("pytest")
