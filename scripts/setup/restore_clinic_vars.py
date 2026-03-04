import paramiko

VPS_IP = "173.214.172.254"
VPS_USER = "root"
VPS_PASS = "4uF&V@FJ"

def restore_clinic_env_vars():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(VPS_IP, username=VPS_USER, password=VPS_PASS, timeout=10)
        print("✅ Connected to VPS.\n")
        
        # Read current .env file
        print("=== 1. Checking current .env file ===")
        stdin, stdout, stderr = client.exec_command(
            "grep CLINIC /opt/heal-query-hub/backend/.env || echo 'No CLINIC vars found'",
            timeout=10
        )
        stdout.channel.recv_exit_status()
        current_env = stdout.read().decode()
        print(current_env)
        
        # Add the clinic API variables
        print("\n=== 2. Adding CLINIC environment variables ===")
        env_additions = '''
# Clinic API Configuration
CLINIC_PROVIDER_LIST_URL=http://41.32.47.162:9091/api/clinicProviderlist/
CLINIC_PROVIDER_SCHEDULE_URL=http://41.32.47.162:9090/api/clinicProviderschedule/
CLINIC_SERVICE_PRICE_URL=http://41.32.47.162:9093/api/servicePrice/
CLINIC_API_USERNAME=millen
CLINIC_API_PASSWORD=millen@4321
REQUEST_TIMEOUT=30
MAX_RETRIES=3
REQUEST_RETRY_BACKOFF=0.5
'''
        
        # Append to .env file
        stdin, stdout, stderr = client.exec_command(
            f"cd /opt/heal-query-hub/backend && echo '{env_additions}' >> .env",
            timeout=10
        )
        stdout.channel.recv_exit_status()
        print("✅ Environment variables added to .env file")
        
        # Verify
        print("\n=== 3. Verifying .env file ===")
        stdin, stdout, stderr = client.exec_command(
            "grep CLINIC /opt/heal-query-hub/backend/.env",
            timeout=10
        )
        stdout.channel.recv_exit_status()
        print(stdout.read().decode())
        
        # Restart containers to load new env vars
        print("\n=== 4. Restarting containers ===")
        stdin, stdout, stderr = client.exec_command(
            "cd /opt/heal-query-hub && docker-compose restart mcp-server backend",
            timeout=60
        )
        stdout.channel.recv_exit_status()
        print(stdout.read().decode())
        print(stderr.read().decode())
        
        # Wait a bit for containers to start
        print("\nWaiting 10 seconds for containers to initialize...")
        import time
        time.sleep(10)
        
        # Test MCP /providers endpoint
        print("\n=== 5. Testing MCP /providers endpoint ===")
        stdin, stdout, stderr = client.exec_command(
            "curl -s http://localhost:8020/providers | head -200",
            timeout=10
        )
        stdout.channel.recv_exit_status()
        result = stdout.read().decode()
        if "success" in result and "clinicId" in result:
            print("✅ MCP IS WORKING!")
            print(result[:500])
        else:
            print("⚠️  MCP still not working:")
            print(result[:500])

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()

if __name__ == "__main__":
    restore_clinic_env_vars()
