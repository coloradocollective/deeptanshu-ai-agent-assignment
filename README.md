# Repository Discovery Agent

An AI Agent that can answer questions about GitHub repositories for a user or an organization.

## Create your repository

1.  Create a new repository using _coloradocollective/ai-agent-assignment_ as a template.
1.  [Create an OpenAI API Key](https://platform.openai.com/settings/organization/api-keys) and record the value.
1.  [Add a repository secret](https://docs.github.com/en/actions/security-for-github-actions/security-guides/using-secrets-in-github-actions)
    named `OPEN_AI_KEY` and set your key from the previous step as the value.
1.  [Create a GitHub token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic)
    with repository access and record the value.
1.  In GitHub actions start the test workflow to see that it passes.
1.  Clone your repository locally.

## Run

1.  [Install uv](https://docs.astral.sh/uv/getting-started/installation/).

1.  Copy the example environment file and fill in the necessary values.
    ```shell
    cp .env.example .env 
    source .env
    ```

1.  Run the app then visit [localhost:5050](http://localhost:5050).
    ```shell
    uv run -m discovery

1.  Paste your GitHub token into the login screen and ask a few queries.

## Test

1.  Run fast tests
    ```shell
    uv run -m unittest
    ```

1.  Run slow tests
    ```shell
    source .env
    RUN_SLOW_TESTS=true uv run -m unittest
    ```

# Exercise

## Create a tool

Use the [GitHub API documentation](https://docs.github.com/en/rest) and add a new tool.

1.  In the [GithubClient](./discovery/github_support/github_client.py), define a method that calls the GitHub API
    endpoint you've chosen.
1.  Add a function to [github_tools](./discovery/repository_agent/github_tools.py) that calls your new method in the
    GitHub client.
    Return a JSON string of the data returned by the method.
1.  Add a Python docstring to your function that describes how OpenAI should use the function.
1.  Add the `@tool()` decorator to the function.
1.  Register the new tool by adding it to the list of tools that are returned by the `github_tools` method.

Run the application and see if you can have OpenAI use your tool to fetch data.

## Add a test for the new tool

Now make sure the agent integrates properly with OpenAI to use your tool.
This test will use the LLM and take more time to run.
The focus of the test should be to check that the agent uses the correct tool to answer the question, but we have
limited ability to check that the response is correct.

1.  Define a `test_` method  in [test_repository_agent.py](./tests/repository_agent/test_repository_agent.py) that will
    cover the new tool that was added.
1.  Decorate the test method with the `@slow` decorator.
1.  Add the `@responses.activate` decorator to the method to enable stubbing the API call.
1.  Stub the API that your tool uses.
1.  Send a question to the agent and assert the correct tools are called
1.  Next, assert the response by spot-checking for relevant word(s).

## Push to GitHub

1.  Make a commit and push your changes to the GitHub repository.
1.  Check to see that your test pass in the GitHub action, then submit your repository URL to Canvas.
