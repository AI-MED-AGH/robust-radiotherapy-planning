# CONTRIBUTION

Some rules should be followed when contributing to this repository:

1. The one who merges a PR is always the person who opened it. This should be done with a merge commit.

2. The first issue in this repository serves as a style guide (#1): it should have a natural language description along with a list of necessary changes in the imperative.

3. The first PR in this repository serves as a style guide (#2): the PR description should be in the present tense, while the description of the merge commit should be written in the imperative.

4. Every commit summary must be written in the imperative starting with a capital letter and including the issue number in parentheses.

5. There always needs to be a corresponding issue before commits are created.

6. All changes to `dev` and `main` must be made through PRs.

7. All PRs going into `dev` or `main` must have approval from a reviewer with write access and pass the code quality workflow.

8. Changing `main` is only done through a PR from `dev`. It follows the same rules as PRs from feature branches, but HEAD must be tagged with a version number. That version number should also be updated in `pyproject.toml`.

9. Force pushing is allowed on feature branches, but should be done with caution, especially if that branch is shared with someone else.

10. Commits should be atomic, but must not break existing functionality.

11. Commits should have a description if the summary is not sufficiently clear.

12. The reviewer should not only look at the changes made in a PR, but also whether commits and the PR meet the standards described here.
