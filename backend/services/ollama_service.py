import asyncio
import os
import platform
import shutil
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, AsyncGenerator
import logging
import signal
import httpx

try:
    import ollama
    HAS_OLLAMA_LIB = True
except ImportError:
    HAS_OLLAMA_LIB = False

logger = logging.getLogger(__name__)

# Get Ollama URL from environment
# Supports: Docker container on same host, remote server, or local install
OLLAMA_HOST = os.getenv('OLLAMA_BASE_URL', os.getenv('OLLAMA_HOST', 'http://localhost:11434'))

class OllamaService:
    """Service for managing Ollama integration - supports local and remote instances."""
    
    def __init__(self):
        self._download_progress = {}
        self._download_processes = {}
        self._host = OLLAMA_HOST
        
        # Initialize clients with configured host
        if HAS_OLLAMA_LIB:
            self._async_client = ollama.AsyncClient(host=self._host)
            self._sync_client = ollama.Client(host=self._host)
        else:
            self._async_client = None
            self._sync_client = None
        
        logger.info(f"OllamaService initialized with host: {self._host}")

    def _is_real_ollama_host(self) -> bool:
        """False when the host points at the LiteLLM proxy (no Ollama /api/tags)."""
        host = self._host or ""
        return bool(host) and "litellm" not in host.lower()

    # =============================================================================
    # PUBLIC API METHODS
    # =============================================================================
    
    async def check_ollama_status(self) -> Dict[str, any]:
        """Check Ollama installation and server status."""
        try:
            is_running = await self._check_server_running()
            # If remote server is reachable, consider it "installed"
            is_installed = is_running or await self._check_installation()
            models, server_url = await self._get_server_info(is_running)
            
            status = {
                "installed": is_installed,
                "running": is_running,
                "server_running": is_running,
                "available_models": models,
                "server_url": server_url or self._host,
                "error": None
            }
            
            logger.debug(f"Ollama status: installed={is_installed}, running={is_running}, models={len(models)}, host={self._host}")
            return status
            
        except Exception as e:
            logger.error(f"Error checking Ollama status: {e}")
            return self._create_error_status(str(e))
    
    async def start_server(self) -> Dict[str, any]:
        """Start the Ollama server (only works for local installs)."""
        try:
            # Check if remote server is already running
            if await self._check_server_running():
                return {"success": True, "message": f"Ollama server is already running at {self._host}"}
            
            # Try to start locally
            success = await self._execute_server_start()
            message = "Ollama server started successfully" if success else "Failed to start Ollama server. If using a remote/Docker instance, ensure the container is running."
            return {"success": success, "message": message}
                
        except Exception as e:
            logger.error(f"Error starting Ollama server: {e}")
            return {"success": False, "message": f"Error starting server: {str(e)}"}
    
    async def stop_server(self) -> Dict[str, any]:
        """Stop the Ollama server (only works for local installs)."""
        try:
            if not await self._check_server_running():
                return {"success": True, "message": "Ollama server is already stopped"}
            
            # For remote/Docker instances, we can't stop them
            if self._host != 'http://localhost:11434' and self._host != 'http://127.0.0.1:11434':
                return {"success": False, "message": f"Cannot stop remote Ollama server at {self._host}. Manage the Docker container directly."}
            
            success = await self._execute_server_stop()
            message = "Ollama server stopped successfully" if success else "Failed to stop Ollama server"
            return {"success": success, "message": message}
                
        except Exception as e:
            logger.error(f"Error stopping Ollama server: {e}")
            return {"success": False, "message": f"Error stopping server: {str(e)}"}
    
    async def download_model(self, model_name: str) -> Dict[str, any]:
        """Download an Ollama model."""
        try:
            success = await self._execute_model_download(model_name)
            message = f"Model {model_name} downloaded successfully" if success else f"Failed to download model {model_name}"
            return {"success": success, "message": message}
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {e}")
            return {"success": False, "message": f"Error downloading model: {str(e)}"}
    
    async def download_model_with_progress(self, model_name: str) -> AsyncGenerator[str, None]:
        """Download an Ollama model with progress streaming."""
        async for progress_data in self._stream_model_download(model_name):
            yield progress_data
    
    async def delete_model(self, model_name: str) -> Dict[str, any]:
        """Delete an Ollama model."""
        try:
            success = await self._execute_model_deletion(model_name)
            message = f"Model {model_name} deleted successfully" if success else f"Failed to delete model {model_name}"
            return {"success": success, "message": message}
        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {e}")
            return {"success": False, "message": f"Error deleting model: {str(e)}"}
    
    async def get_recommended_models(self) -> List[Dict[str, str]]:
        """Get list of recommended Ollama models."""
        try:
            models_path = self._get_ollama_models_path()
            if models_path.exists():
                return self._load_models_from_file(models_path)
            else:
                return self._get_fallback_models()
        except Exception as e:
            logger.error(f"Error loading recommended models: {e}")
            return self._get_fallback_models()
    
    async def get_available_models(self) -> List[Dict[str, str]]:
        """Get available Ollama models formatted for the language models API."""
        try:
            status = await self.check_ollama_status()
            if not status.get("running", False):
                return []
            
            downloaded_models = status.get("available_models", [])
            if not downloaded_models:
                return []
            
            api_models = self._format_models_for_api(downloaded_models)
            logger.debug(f"Returning {len(api_models)} Ollama models for API")
            return api_models
            
        except Exception as e:
            logger.error(f"Error getting available models for API: {e}")
            return []
    
    def get_download_progress(self, model_name: str) -> Optional[Dict[str, any]]:
        return self._download_progress.get(model_name)
    
    def get_all_download_progress(self) -> Dict[str, Dict[str, any]]:
        return self._download_progress.copy()
    
    def cancel_download(self, model_name: str) -> bool:
        if model_name in self._download_progress:
            self._download_progress[model_name] = {
                "status": "cancelled",
                "message": f"Download of {model_name} was cancelled",
                "error": "Download cancelled by user"
            }
            return True
        return False
    
    # =============================================================================
    # PRIVATE HELPER METHODS
    # =============================================================================

    def _create_error_status(self, error: str) -> Dict[str, any]:
        return {
            "installed": False,
            "running": False,
            "server_running": False,
            "available_models": [],
            "server_url": self._host,
            "error": error
        }
    
    async def _check_installation(self) -> bool:
        """Check if Ollama is installed locally OR available remotely."""
        # Skip remote probe when the host is the LiteLLM proxy (not Ollama).
        if not self._is_real_ollama_host():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._is_ollama_installed)
        # First check if remote server is reachable
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self._host}/api/tags")
                if resp.status_code == 200:
                    return True
        except Exception:
            pass
        
        # Fall back to checking local installation
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._is_ollama_installed)
    
    def _is_ollama_installed(self) -> bool:
        # shutil.which handles PATH lookup across platforms without spawning a shell.
        try:
            return shutil.which("ollama") is not None
        except Exception:
            return False
    
    async def _check_server_running(self) -> bool:
        """Check if the Ollama server is running (local or remote)."""
        # The LiteLLM proxy is not an Ollama server; skip probing it.
        if not self._is_real_ollama_host():
            return False
        # Try with ollama client first
        if self._async_client:
            try:
                await asyncio.wait_for(self._async_client.list(), timeout=2.0)
                logger.debug(f"Ollama server confirmed running at {self._host}")
                return True
            except Exception:
                pass
        
        # Fall back to direct HTTP check
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._host}/api/tags")
                if resp.status_code == 200:
                    logger.debug(f"Ollama server confirmed running via HTTP at {self._host}")
                    return True
        except Exception as e:
            logger.debug(f"Ollama server not reachable at {self._host}: {e}")
        
        return False
    
    async def _get_server_info(self, is_running: bool) -> tuple:
        """Get server information (models and URL) if server is running."""
        if not is_running:
            return [], ""
        
        # Try with ollama client
        if self._async_client:
            try:
                response = await self._async_client.list()
                models = [model.model for model in response.models]
                logger.debug(f"Found {len(models)} models via ollama client")
                return models, self._host
            except Exception as e:
                logger.debug(f"Failed to list via ollama client: {e}")
        
        # Fall back to direct HTTP
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._host}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m.get('name', m.get('model', '')) for m in data.get('models', [])]
                    logger.debug(f"Found {len(models)} models via HTTP")
                    return models, self._host
        except Exception as e:
            logger.debug(f"Failed to get server info via HTTP: {e}")
        
        return [], self._host
    
    async def _execute_server_start(self) -> bool:
        if self._sync_client:
            try:
                self._sync_client.list()
                return True
            except Exception:
                pass
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._start_ollama_process)
    
    def _start_ollama_process(self) -> bool:
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return self._wait_for_server_start()
        except Exception as e:
            logger.error(f"Error starting Ollama server: {e}")
            return False
    
    def _wait_for_server_start(self) -> bool:
        for i in range(20):
            time.sleep(1)
            try:
                if self._sync_client:
                    self._sync_client.list()
                    logger.info(f"Ollama server started after {i+1}s")
                    return True
            except Exception:
                continue
        return False
    
    async def _execute_server_stop(self) -> bool:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._stop_ollama_process)
    
    def _stop_ollama_process(self) -> bool:
        system = platform.system().lower()
        try:
            if system in ["darwin", "linux"]:
                result = subprocess.run(["pgrep", "-f", "ollama serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode == 0:
                    pids = [pid for pid in result.stdout.strip().split('\n') if pid]
                    for pid in pids:
                        try:
                            os.kill(int(pid), signal.SIGTERM)
                        except Exception:
                            pass
            time.sleep(2)
            return True
        except Exception as e:
            logger.error(f"Error stopping Ollama: {e}")
            return False
    
    async def _execute_model_download(self, model_name: str) -> bool:
        if not await self._check_server_running():
            logger.error(f"Cannot download {model_name}: server not running")
            return False
        try:
            if self._async_client:
                await self._async_client.pull(model_name)
                return True
            # Fallback: HTTP pull
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(f"{self._host}/api/pull", json={"name": model_name})
                return resp.status_code == 200
        except Exception as e:
            logger.error(f"Error downloading {model_name}: {e}")
            return False
    
    async def _execute_model_deletion(self, model_name: str) -> bool:
        if not await self._check_server_running():
            return False
        try:
            if self._async_client:
                await self._async_client.delete(model_name)
                return True
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.delete(f"{self._host}/api/delete", json={"name": model_name})
                return resp.status_code == 200
        except Exception as e:
            logger.error(f"Error deleting {model_name}: {e}")
            return False
    
    async def _stream_model_download(self, model_name: str) -> AsyncGenerator[str, None]:
        try:
            if not await self._check_server_running():
                yield f'data: {json.dumps({"status": "error", "error": "Ollama server is not running"})}\n\n'
                return
            
            self._download_progress[model_name] = {"status": "starting", "percentage": 0}
            yield f'data: {json.dumps({"status": "starting", "percentage": 0, "message": f"Starting download of {model_name}..."})}\n\n'
            
            if self._async_client:
                pull_stream = await self._async_client.pull(model_name, stream=True)
                async for progress in pull_stream:
                    progress_data = self._process_download_progress(progress, model_name)
                    if progress_data:
                        yield f'data: {json.dumps(progress_data)}\n\n'
                        if progress_data.get("status") == "completed":
                            break
            else:
                # HTTP fallback
                async with httpx.AsyncClient(timeout=600.0) as client:
                    async with client.stream("POST", f"{self._host}/api/pull", json={"name": model_name}) as resp:
                        async for line in resp.aiter_lines():
                            if line:
                                try:
                                    data = json.loads(line)
                                    pct = 0
                                    if data.get('total', 0) > 0 and data.get('completed', 0) > 0:
                                        pct = (data['completed'] / data['total']) * 100
                                    pd = {"status": data.get('status', 'downloading'), "percentage": pct, "message": data.get('status', '')}
                                    self._download_progress[model_name] = pd
                                    yield f'data: {json.dumps(pd)}\n\n'
                                except json.JSONDecodeError:
                                    pass
                
                final = {"status": "completed", "percentage": 100, "message": f"Model {model_name} downloaded!"}
                self._download_progress[model_name] = final
                yield f'data: {json.dumps(final)}\n\n'
                    
        except Exception as e:
            error_data = {"status": "error", "message": str(e), "error": str(e)}
            self._download_progress[model_name] = error_data
            yield f'data: {json.dumps(error_data)}\n\n'
        finally:
            await asyncio.sleep(1)
            self._download_progress.pop(model_name, None)
    
    def _process_download_progress(self, progress, model_name: str) -> Optional[Dict[str, any]]:
        if not hasattr(progress, 'status'):
            return None
        
        progress_data = {"status": "downloading", "message": progress.status}
        
        if (hasattr(progress, 'completed') and hasattr(progress, 'total') and
            progress.total and progress.completed is not None and progress.total > 0):
            pct = (progress.completed / progress.total) * 100
            progress_data.update({"percentage": pct, "bytes_downloaded": progress.completed, "total_bytes": progress.total})
        
        if hasattr(progress, 'digest'):
            progress_data["digest"] = progress.digest
        
        self._download_progress[model_name] = progress_data
        
        if progress.status == "success" or (
            hasattr(progress, 'completed') and hasattr(progress, 'total') and
            progress.completed is not None and progress.total is not None and
            progress.completed == progress.total and progress.total > 0):
            final = {"status": "completed", "percentage": 100, "message": f"Model {model_name} downloaded!"}
            self._download_progress[model_name] = final
            return final
        
        return progress_data
    
    def _get_ollama_models_path(self) -> Path:
        return Path(__file__).parent.parent.parent.parent / "src" / "llm" / "ollama_models.json"
    
    def _load_models_from_file(self, models_path: Path) -> List[Dict[str, str]]:
        with open(models_path, 'r') as f:
            return json.load(f)
    
    def _get_fallback_models(self) -> List[Dict[str, str]]:
        return [
            {"display_name": "[alibaba] qwen2.5 (14B)", "model_name": "qwen2.5:14b", "provider": "Ollama"},
            {"display_name": "[microsoft] phi4 (14B)", "model_name": "phi4:latest", "provider": "Ollama"},
            {"display_name": "[microsoft] phi3.5 (3.8B)", "model_name": "phi3.5:latest", "provider": "Ollama"},
            {"display_name": "[meta] llama3.1 (8B)", "model_name": "llama3.1:latest", "provider": "Ollama"},
            {"display_name": "[google] gemma3 (4B)", "model_name": "gemma3:4b", "provider": "Ollama"},
        ]
    
    def _format_models_for_api(self, downloaded_models: List[str]) -> List[Dict[str, str]]:
        """Format downloaded models for API response."""
        api_models = []
        for model_name in downloaded_models:
            # Clean up model name for display
            model_name.split(':')[0] if ':' in model_name else model_name
            api_models.append({
                "display_name": f"[ollama] {model_name}",
                "model_name": model_name,
                "provider": "Ollama"
            })
        return api_models


# Global service instance
ollama_service = OllamaService()
