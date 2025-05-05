import unittest

import responses
from openai import OpenAI

from discovery.agent_support.agent import ToolCall
from discovery.environment import require_env
from discovery.github_support.github_client import GithubClient
from discovery.repository_agent.repository_agent import repository_agent_creator
from tests.slow_test_support import slow


class TestRepositoryAgent(unittest.TestCase):
    @slow
    def setUp(self):
        openai_client = OpenAI(api_key=require_env("OPEN_AI_KEY"))
        self.repo_dict = {
            "name": "pickles_web",
            "full_name": "pickles_company/pickles_web",
            "html_url": "https://example.com/pickles_company/pickles_web",
            "url": "https://api.example.com/pickles_company/pickles_web",
            "description": "Pickles Web",
            "private": False,
            "stargazers_count": 12,
            "watchers_count": 13,
            "forks_count": 14,
        }

        self.agent = repository_agent_creator(openai_client)(GithubClient("some-token"))

    @slow
    @responses.activate
    def test_list_repositories_for_organization(self):
        responses.add(
            responses.GET,
            "https://api.github.com/orgs/pickles_company/repos",
            json=[self.repo_dict],
            status=200,
        )

        result = self.agent.answer("List repositories for the organization pickles_company.")

        self.assertIn("pickles_web", result.response)
        self.assertEqual([ToolCall(name="list_repositories_for_organization", arguments={
            "organization": "pickles_company"
        })], result.tool_calls)

    @slow
    @responses.activate
    def test_list_repositories_for_user(self):
        responses.add(
            responses.GET,
            "https://api.github.com/users/some_user/repos",
            json=[self.repo_dict],
            status=200,
        )

        result = self.agent.answer("List repositories for some_user.")

        self.assertIn("pickles_web", result.response)
        self.assertEqual([ToolCall(name="list_repositories_for_user", arguments={
            "user": "some_user"
        })], result.tool_calls)

    @slow
    @responses.activate
    def test_search_repositories(self):
        responses.add(
            responses.GET,
            "https://api.github.com/search/repositories",
            json={
                "items": [
                    {
                        "name": "pickles_web",
                        "full_name": "pickles_company/pickles_web",
                        "html_url": "https://example.com/pickles_company/pickles_web",
                        "url": "https://api.example.com/pickles_company/pickles_web",
                        "description": "Pickles Web",
                        "private": False,
                        "stargazers_count": 12,
                        "watchers_count": 13,
                        "forks_count": 14,
                    },
                    {
                        "name": "chickens_web",
                        "full_name": "pickles_company/dill_pickles_web",
                        "html_url": "https://example.com/pickles_company/dill_pickles_web",
                        "url": "https://api.example.com/pickles_company/dill_pickles_web",
                        "description": "Dill Pickles Web",
                        "private": True,
                        "stargazers_count": 12,
                        "watchers_count": 13,
                        "forks_count": 15
                    }
                ]
            },
            status=200,
        )

        result = self.agent.answer(
            "Search the repositories that are private for the pickles_company and tell me the name of the repository")

        self.assertIn("dill_pickles_web", result.response)
        self.assertEqual(1, len(result.tool_calls))
        self.assertEqual('pickles_company', result.tool_calls[0].arguments['owner'])
        self.assertEqual('org', result.tool_calls[0].arguments['owner_type'])

    @slow
    @responses.activate
    def test_list_repository_languages(self):
        responses.add(
            responses.GET,
            "https://api.github.com/repos/pickles_org/pickles_repo/languages",
            json={
                "Python": 100,
                "HTML": 50,
                "CSS": 50,
            },
            status=200,
        )

        result = self.agent.answer("What languages does the 'pickles_repo' from the 'pickles_org' use?")

        self.assertIn("Python", result.response)
        self.assertIn("HTML", result.response)
        self.assertIn("CSS", result.response)
        self.assertNotIn("java", result.response)
        self.assertEqual(
            [ToolCall(name='list_repository_languages', arguments={'full_name': 'pickles_org/pickles_repo'})],
            result.tool_calls
        )

    @slow
    @responses.activate
    def test_list_repository_contributors(self):
        responses.add(
            responses.GET,
            "https://api.github.com/repos/pickles_org/pickles_repo/contributors",
            json=[
                {"login": "Fred"},
                {"login": "Mary"},
                {"login": "Kate"},
            ],
            status=200,
        )

        result = self.agent.answer("Who contributes to the pickles_repo within the pickles_org?")

        self.assertIn("Fred", result.response)
        self.assertIn("Mary", result.response)
        self.assertIn("Kate", result.response)
        self.assertEqual(
            [ToolCall(
                name='list_repository_contributors',
                arguments={'full_name': 'pickles_org/pickles_repo'}
            )],
            result.tool_calls
        )

    @slow
    @responses.activate
    def test_list_repository_issues(self):
        responses.add(
            responses.GET,
            "https://api.github.com/repos/pickles_org/pickles_repo/issues",
            json=[
                {
                    "number": 1,
                    "title": "Fix login bug",
                    "state": "open",
                    "html_url": "https://github.com/pickles_org/pickles_repo/issues/1",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-02T00:00:00Z",
                    "body": "The login page is not working properly",
                    "user": {"login": "Alice"}
                },
                {
                    "number": 2,
                    "title": "Add new feature",
                    "state": "open",
                    "html_url": "https://github.com/pickles_org/pickles_repo/issues/2",
                    "created_at": "2024-01-03T00:00:00Z",
                    "updated_at": "2024-01-04T00:00:00Z",
                    "body": "We need to add a new feature",
                    "user": {"login": "Bob"}
                }
            ],
            status=200,
        )

        result = self.agent.answer("What are the open issues in the pickles_repo from pickles_org?")

        self.assertIn("Fix login bug", result.response)
        self.assertIn("Add new feature", result.response)
        self.assertIn("Alice", result.response)
        self.assertIn("Bob", result.response)
        self.assertEqual(
            [ToolCall(
                name='list_repository_issues',
                arguments={'full_name': 'pickles_org/pickles_repo', 'state': 'open'}
            )],
            result.tool_calls
        )
