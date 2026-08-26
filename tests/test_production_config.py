"""Tests that preserve private and scoped production deployment settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _compose_config() -> dict[str, Any]:
    """Load the production Compose file as validated YAML data."""
    compose_path = PROJECT_ROOT / "docker-compose.prod.yml"
    payload = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_production_service_is_private_and_uses_shared_network() -> None:
    """The AI container has no host port and joins the external network."""
    compose = _compose_config()
    service = compose["services"]["ai"]
    network = compose["networks"]["tattoo_hysteria_net"]

    assert service["container_name"] == "tattoo_hysteria_ai"
    assert service["expose"] == ["8001"]
    assert "ports" not in service
    assert service["networks"] == ["tattoo_hysteria_net"]
    assert network["external"] is True
    assert network["name"] == "tattoo_hysteria_net"


def test_production_container_runs_as_non_root() -> None:
    """The image uses a non-root user and the required bind address."""
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "USER app" in dockerfile
    assert '"--host", "0.0.0.0"' in dockerfile
    assert '"--port", "8001"' in dockerfile
    assert "api.main:app" in dockerfile


def test_deployment_script_is_scoped_and_avoids_global_cleanup() -> None:
    """Deployment remains inside the AI folder and avoids global pruning."""
    script_path = PROJECT_ROOT / "scripts" / "deploy-production.sh"
    script = script_path.read_text(encoding="utf-8")

    assert '/opt/tattoo-hysteria-ai' in script
    assert 'tattoo_hysteria_net' in script
    assert "docker system prune" not in script
    assert "/opt/tattoo-hysteria-backend" not in script


def test_workflow_provisions_environment_without_printing_secret() -> None:
    """CI writes the server environment atomically through encrypted SSH."""
    workflow_path = PROJECT_ROOT / ".github" / "workflows" / "production.yml"
    workflow = workflow_path.read_text(encoding="utf-8")

    assert "OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}" in workflow
    assert "cat > /opt/tattoo-hysteria-ai/.env.next" in workflow
    assert "chmod 600 /opt/tattoo-hysteria-ai/.env.next" in workflow
    assert "mv /opt/tattoo-hysteria-ai/.env.next" in workflow
    assert 'echo "${OPENAI_API_KEY}"' not in workflow


def test_workflow_bootstraps_checkout_without_ec2_github_credentials() -> None:
    """CI sends the exact Git commit and creates the missing AI directory."""
    workflow_path = PROJECT_ROOT / ".github" / "workflows" / "production.yml"
    workflow = workflow_path.read_text(encoding="utf-8")

    assert "git bundle create" in workflow
    assert "sudo install -d" in workflow
    assert "git clone \"${BUNDLE_PATH}\"" in workflow
    assert "git fetch origin main" not in workflow
    assert "git pull" not in workflow

