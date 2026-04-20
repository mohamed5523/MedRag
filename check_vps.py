"""Quick diagnostic script for VPS issues."""
import sys
sys.path.insert(0, '/home/morad/Projects/heal-query-hub/scripts/deploy')
import paramiko

VPS_IP = "173.214.172.254"
VPS_USER = "root"
VPS_PASS = "4uF&V@FJ"

def ssh_run(client, cmd, timeout=15):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    stdout.channel.settimeout(timeout)
    try:
        out = stdout.read().decode(errors='replace')
        err = stderr.read().decode(errors='replace')
        return out, err
    except Exception as e:
        return f"TIMEOUT: {e}", ""

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(VPS_IP, username=VPS_USER, password=VPS_PASS, timeout=10)
print("Connected to VPS!")

print("\n=== BACKEND LOGS (last 30 lines) ===")
out, err = ssh_run(client, "docker logs heal-query-hub-backend-1 2>&1 | tail -30")
print(out or err)

print("\n=== HOST NGINX SITE CONFIG ===")
out, err = ssh_run(client, "cat /etc/nginx/sites-enabled/*")
print(out or err)

print("\n=== NGINX ERROR LOG (last 20) ===")
out, err = ssh_run(client, "tail -20 /var/log/nginx/error.log 2>/dev/null || echo 'no error log'")
print(out or err)

print("\n=== NGINX ACCESS LOG (last 10 /api/ hits) ===")
out, err = ssh_run(client, "grep '/api/' /var/log/nginx/access.log 2>/dev/null | tail -10 || echo 'no /api/ entries'")
print(out or err)

print("\n=== DOCKER COMPOSE ON VPS ===")
out, err = ssh_run(client, "cd /opt/heal-query-hub && docker ps --format 'table {{.Names}}\t{{.Status}}'")
print(out or err)

print("\n=== TEST DIRECT to port 8080 ===")
out, err = ssh_run(client, "curl -s -o /dev/null -w '%{http_code} %{time_total}s' --max-time 5 http://localhost:8080/api/chat/session/new -X POST", timeout=10)
print(out or err)

client.close()
print("\nDone!")
