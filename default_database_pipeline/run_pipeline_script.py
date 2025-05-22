# coding:utf-8
from default_database_pipeline import run_pipeline
import getpass

if __name__ == "__main__":
    username = input("Enter your WRDS username: ")
    password = getpass.getpass("Enter your WRDS password: ")
    run_pipeline(username, password)