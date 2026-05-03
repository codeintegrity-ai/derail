"""Shared pytest config for derail tests.

``asyncio_mode = "auto"`` in pyproject.toml does the heavy lifting — every
``async def test_*`` is auto-run by pytest-asyncio without needing explicit
``@pytest.mark.asyncio`` decorators.
"""
