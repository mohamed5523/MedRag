import paramiko
import os
import zipfile
import re
import sys

VPS_IP = "173.214.172.254"
VPS_USER = "root"
VPS_PASS = "4uF&V@FJ"
PROJECT_DIR = "/home/morad/Projects/heal-query-hub"
REMOTE_DIR = "/opt/heal-query-hub"

# Keys to extract from different files
BACKEND_KEYS = [
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "TTS_PROVIDER",
    "OPENAI_TTS_MODEL",
    "OPENAI_TTS_VOICE",
    "OPENAI_TTS_AUDIO_FORMAT",
    "ELEVENLABS_API_KEY",
    "ELEVENLABS_VOICE_ID",
    "ELEVENLABS_MODEL",
    "ELEVENLABS_STABILITY",
    "ELEVENLABS_SIMILARITY_BOOST",
    "ELEVENLABS_STYLE",
    "ELEVENLABS_OUTPUT_FORMAT",
    "AZURE_TTS_API_KEY",
    "AZURE_TTS_ENDPOINT",
    "AZURE_TTS_REGION",
    "AZURE_TTS_OUTPUT_FORMAT",
    "AZURE_TTS_VOICE",
    "WHATSAPP_TOKEN",
    "WHATSAPP_PHONE_NUMBER_ID",
    "WHATSAPP_VERIFY_TOKEN",
    "DEFAULT_TZ",
    "CHROMA_DB_PATH",
    "UPLOAD_DIR",
    "MAX_FILE_SIZE_MB",
    "LOG_LEVEL",
    "CLINIC_PROVIDER_LIST_URL",
    "CLINIC_PROVIDER_SCHEDULE_URL",
    "CLINIC_SERVICE_PRICE_URL",
    "CLINIC_API_USERNAME",
    "CLINIC_API_PASSWORD",
    "REQUEST_TIMEOUT",
    "MAX_RETRIES",
    "REQUEST_RETRY_BACKOFF"
]

FRONTEND_KEYS = [
    "VITE_SUPABASE_URL",
    "VITE_SUPABASE_ANON_KEY"
]

def parse_env_file(filepath):
    env_vars = {}
    if not os.path.exists(filepath):
        return env_vars
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                value = value.strip('"').strip("'")
                env_vars[key.strip()] = value
    return env_vars

def get_config():
    config = {
        "backend": {},
        "frontend": {}
    }
    
    backend_env_path = os.path.join(PROJECT_DIR, "backend", ".env.previous")
    if not os.path.exists(backend_env_path):
         backend_env_path = os.path.join(PROJECT_DIR, "backend", ".env")
    
    backend_vars = parse_env_file(backend_env_path)
    for key in BACKEND_KEYS:
        if key in backend_vars:
            config["backend"][key] = backend_vars[key]

    config["backend"]["API_HOST"] = "0.0.0.0"
    config["backend"]["API_PORT"] = "8000"
    config["backend"]["WEAVIATE_URL"] = "http://weaviate:8080"
    config["backend"]["REDIS_HOST"] = "redis"
    config["backend"]["REDIS_PORT"] = "6379"
    config["backend"]["CORS_ORIGINS"] = "*"

    root_env_path = os.path.join(PROJECT_DIR, ".env")
    root_vars = parse_env_file(root_env_path)
    for key in FRONTEND_KEYS:
        if key in root_vars:
            config["frontend"][key] = root_vars[key]
            
    config["frontend"]["VITE_API_URL"] = f"https://dsb-kairo.de/api"
    
    return config

def create_ssh_client():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(VPS_IP, username=VPS_USER, password=VPS_PASS)
    client.get_transport().set_keepalive(30)  # Send keepalive every 30s to prevent timeouts
    return client

def run_command(client, command, stream_output=False):
    print(f"Running: {command}")
    stdin, stdout, stderr = client.exec_command(command, get_pty=stream_output)
    
    if stream_output:
        for line in iter(stdout.readline, ""):
            print(line, end="")
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            err = stderr.read().decode()
            print(f"Error ({exit_status}): {err}")
            return False, err
        return True, ""
    else:
        exit_status = stdout.channel.recv_exit_status()
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        
        if exit_status != 0:
            print(f"Error ({exit_status}): {err}")
            return False, err
        if out:
            print(out)
        return True, out

def write_remote_file(client, path, content):
    print(f"Writing config to {path}...")
    content_escaped = content.replace("'", "'\\''")
    command = f"cat > '{path}' << 'EOF'\n{content}\nEOF"
    success, _ = run_command(client, command)
    return success

def zip_project(zip_filename):
    print("Zipping project files...")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(PROJECT_DIR):
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.venv', 'venv', 'deploy_venv', 'dist', 'build']]
            for file in files:
                if file == zip_filename or file.endswith('.pyc') or file.endswith('.zip'):
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, PROJECT_DIR)
                zipf.write(file_path, arcname)

def main():
    zip_filename = "project_deploy.zip"
    
    print("Preparing deployment...")
    config = get_config()
    
    print(f"Connecting to {VPS_IP}...")
    try:
        client = create_ssh_client()
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    # Upload setup script
    success, path_exists = run_command(client, "[ -f /root/setup_vps.sh ] && echo yes || echo no")
    if "no" in path_exists: # Initial setup
        sftp = client.open_sftp()
        sftp.put("setup_vps.sh", "/root/setup_vps.sh")
        sftp.close()
        run_command(client, "chmod +x /root/setup_vps.sh")
        success, _ = run_command(client, "/root/setup_vps.sh", stream_output=True)
        if not success:
            print("VPS setup failed.")
            return

    zip_project(zip_filename)
    
    print("Uploading project code...")
    run_command(client, f"mkdir -p {REMOTE_DIR}")
    
    import time
    for attempt in range(3):
        try:
            sftp = client.open_sftp()
            sftp.put(zip_filename, f"{REMOTE_DIR}/{zip_filename}")
            sftp.close()
            print("Upload successful!")
            break
        except Exception as e:
            print(f"Upload failed (attempt {attempt+1}/3): {e}")
            if attempt < 2:
                print("Retrying in 2 seconds...")
                time.sleep(2)
                # Try to reconnect if connection looks dead
                try:
                    client.close()
                except:
                    pass
                client = create_ssh_client()
            else:
                print("Failed to upload project after 3 attempts.")
                if os.path.exists(zip_filename):
                    os.remove(zip_filename)
                return
    
    print("Extracting files...")
    run_command(client, f"mkdir -p {REMOTE_DIR}/backend {REMOTE_DIR}/frontend")
    run_command(client, f"cd {REMOTE_DIR} && unzip -o {zip_filename}")
    
    # Configure Environment Variables
    backend_env_content = ""
    for k, v in config["backend"].items():
        backend_env_content += f"{k}={v}\n"
    write_remote_file(client, f"{REMOTE_DIR}/backend/.env", backend_env_content)
    
    frontend_env_content = ""
    for k, v in config["frontend"].items():
        frontend_env_content += f"{k}={v}\n"
    write_remote_file(client, f"{REMOTE_DIR}/frontend/.env", frontend_env_content)

    print("Starting application with Docker Compose...")
    # Using stream_output=True to show build progress
    cmd = f"cd {REMOTE_DIR} && /usr/local/bin/docker-compose up -d --build --remove-orphans"
    success, out = run_command(client, cmd, stream_output=True)
    
    if success:
        print("\n✅ Deployment successful!")
        print(f"Frontend: http://{VPS_IP}:8080")
        print(f"Backend Health: http://{VPS_IP}:8000/health")
    else:
        print("\n❌ Deployment failed.")

    if os.path.exists(zip_filename):
        os.remove(zip_filename)
    client.close()

if __name__ == "__main__":
    main()
