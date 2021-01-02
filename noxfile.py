import nox


@nox.session(python="3.9")
def tests(session):
    session.install("-r", "requirements_dev.txt")
    session.run('pytest')
    session.notify("coverage")
