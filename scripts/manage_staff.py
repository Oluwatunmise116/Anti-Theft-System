"""
Manage staff accounts for the gate web app.

Run from the project root (it uses the same database as the app, honouring
LICENSE_DB_PATH from the environment or .env):

    python scripts/manage_staff.py create  <username> --role admin
    python scripts/manage_staff.py create  <username> --role operator
    python scripts/manage_staff.py passwd  <username>
    python scripts/manage_staff.py disable <username>
    python scripts/manage_staff.py enable  <username>
    python scripts/manage_staff.py list

Passwords are read interactively (never from the command line, which would
leave them in shell history).
"""
import argparse
import getpass
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

import auth  # noqa: E402  (after .env: database reads LICENSE_DB_PATH at import)


def _read_password() -> str:
    first = getpass.getpass("Password: ")
    if first != getpass.getpass("Repeat password: "):
        raise ValueError("Passwords do not match.")
    return first


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Manage gate staff accounts.")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("username")
    create.add_argument("--role", choices=auth.ROLES, default="operator")
    for name in ("passwd", "disable", "enable"):
        sub.add_parser(name).add_argument("username")
    sub.add_parser("list")
    args = parser.parse_args(argv)

    try:
        if args.command == "create":
            auth.create_user(args.username, _read_password(), args.role)
            print(f"Created {args.role} account {args.username!r}.")
        elif args.command == "passwd":
            auth.set_password(args.username, _read_password())
            print(f"Password changed for {args.username!r}; existing sessions were signed out.")
        elif args.command in ("disable", "enable"):
            auth.set_active(args.username, args.command == "enable")
            print(f"{args.username!r} {args.command}d.")
        else:
            for user in auth.list_users():
                last = (datetime.fromtimestamp(user["last_login_at"]).strftime("%Y-%m-%d %H:%M")
                        if user["last_login_at"] else "never")
                state = "active" if user["active"] else "disabled"
                print(f"{user['username']:<24} {user['role']:<9} {state:<9} last login {last}")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
