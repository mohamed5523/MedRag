import paramiko
import time

VPS_IP = "173.214.172.254"
VPS_USER = "root"
VPS_PASS = "4uF&V@FJ"

def install_directadmin():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(VPS_IP, username=VPS_USER, password=VPS_PASS)
        print("Connected. Downloading DirectAdmin setup...")
        
        # Download setup.sh
        stdin, stdout, stderr = client.exec_command("wget -O setup.sh https://www.directadmin.com/setup.sh")
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            print(f"Failed to download setup.sh: {stderr.read().decode()}")
            return

        print("Running installation (this may take 20-30 minutes)...")
        # Run setup.sh auto
        # We need a PTY to see output and avoid buffering issues, although auto shouldn't need interaction
        stdin, stdout, stderr = client.exec_command("chmod +x setup.sh && ./setup.sh auto", get_pty=True)
        
        for line in iter(stdout.readline, ""):
            print(line, end="")
            
        exit_status = stdout.channel.recv_exit_status()
        if exit_status == 0:
            print("\nDirectAdmin installation completed successfully!")
        else:
            print(f"\nDirectAdmin installation failed with exit code {exit_status}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    install_directadmin()
