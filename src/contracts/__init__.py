"""The wire contract: request and response models of the REST API.

Pure pydantic - no domain logic, no engine imports - so the HTTP routes and the SDK's thin client share
one definition of every shape. Mapping from domain objects to these models lives in
:mod:`src.application.mappers`. Changes here are changes to the public wire contract: additive only
within ``/api/v1`` (see ``docs/framework.md``).
"""
