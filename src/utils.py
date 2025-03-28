import git

__git_repo = git.Repo(".", search_parent_directories=True)
GIT_ROOT = __git_repo.git.rev_parse("--show-toplevel")

MODELS_PATH = f"{GIT_ROOT}/models"
DATASETS_PATH = f"{GIT_ROOT}/datasets"
