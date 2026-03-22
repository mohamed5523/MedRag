import paramiko

VPS_IP = "173.214.172.254"
VPS_USER = "root"
VPS_PASS = "4uF&V@FJ"

print("Connecting to VPS to clear disk space...")
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(VPS_IP, username=VPS_USER, password=VPS_PASS)

print("Running docker system prune -af --volumes")
stdin, stdout, stderr = client.exec_command("docker system prune -af --volumes")
out = stdout.read().decode()
err = stderr.read().decode()

if "Total reclaimed space" in out or out:
    print(out)
else:
    print(err)

client.close()
print("Disk space cleared.")
