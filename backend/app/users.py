"""Manage users from the command line. (Admins can also do this on the web app's Users page.)

    python -m app.users list
    python -m app.users add NAME        # prompts for the password; --admin to make them an admin
    python -m app.users passwd NAME
    python -m app.users delete NAME     # refused while the user still owns sessions

Pass --password-stdin to read the password from stdin instead of prompting (for scripts).
"""

import argparse
import getpass
import sys

from sqlalchemy import func, select

from app.auth import hash_password, normalize_username, password_problem, username_problem
from app.db import get_sessionmaker
from app.models import RecordingSession, User


def _password(from_stdin: bool) -> str:
    if from_stdin:
        pw = sys.stdin.readline().rstrip("\n")
    else:
        pw = getpass.getpass("password: ")
        if getpass.getpass("again: ") != pw:
            sys.exit("passwords don't match")
    if problem := password_problem(pw):
        sys.exit(problem)
    return pw


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m app.users", description="Manage hit-far users.")
    ap.add_argument("command", choices=["list", "add", "passwd", "delete"])
    ap.add_argument("username", nargs="?")
    ap.add_argument("--password-stdin", action="store_true")
    ap.add_argument("--admin", action="store_true", help="with add: the new user can manage users")
    a = ap.parse_args(argv)
    name = normalize_username(a.username or "")
    if a.command != "list" and not name:
        ap.error(f"{a.command} needs a username")

    with get_sessionmaker()() as db:
        if a.command == "list":
            n = select(func.count(RecordingSession.id)).where(RecordingSession.user_id == User.id).scalar_subquery()
            for u, sessions in db.execute(select(User, n).order_by(User.created_at)).all():
                admin = "\tadmin" if u.is_admin else ""
                print(f"{u.username}\t{sessions} sessions\tcreated {u.created_at:%Y-%m-%d}{admin}")
            return
        user = db.scalar(select(User).where(User.username == name))
        if a.command == "add":
            if problem := username_problem(name):
                sys.exit(problem)
            if user is not None:
                sys.exit(f"user {name!r} already exists")
            db.add(User(username=name, password_hash=hash_password(_password(a.password_stdin)), is_admin=a.admin))
        elif user is None:
            sys.exit(f"no user {name!r}")
        elif a.command == "passwd":
            user.password_hash = hash_password(_password(a.password_stdin))
        elif a.command == "delete":
            owned = db.scalar(select(func.count()).select_from(RecordingSession)
                              .where(RecordingSession.user_id == user.id))
            if owned:
                sys.exit(f"{name!r} still owns {owned} sessions; delete them first")
            db.delete(user)
        db.commit()
        print(f"{a.command}: {name} ok")


if __name__ == "__main__":
    main()
