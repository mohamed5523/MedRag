import paramiko

VPS_IP = "173.214.172.254"
VPS_USER = "root"
VPS_PASS = "4uF&V@FJ"

def update_license():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(VPS_IP, username=VPS_USER, password=VPS_PASS)
        print("Connected. Attempting to refresh license...")
        
        # Try getLicense.sh auto
        # The path usually is /usr/local/directadmin/scripts/getLicense.sh
        
        cmd = "cd /usr/local/directadmin/scripts && ./getLicense.sh auto"
        print(f"Running: {cmd}")
        stdin, stdout, stderr = client.exec_command(cmd)
        
        out = stdout.read().decode()
        err = stderr.read().decode()
        
        print("STDOUT:", out)
        print("STDERR:", err)
        
        # Restart DirectAdmin to apply changes
        print("Restarting DirectAdmin...")
        client.exec_command("service directadmin restart")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    update_license()
