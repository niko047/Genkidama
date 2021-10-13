import json
from channels.generic.websocket import AsyncWebsocketConsumer
import nmap

class ScanComputationalNodes(AsyncWebsocketConsumer):

    # groups = ["broadcast"]

    async def connect(self):
        #Accepts only requests from itself (or very trusted nodes)
        await self.accept()

    async def receive(self, text_data=None, bytes_data=None):

        WS_PORT = '8000'
        NETWORK = '192.168.1.0'

        #Pings the networks and checks for pingable IPs
        nm = nmap.PortScanner()

        #Change it, should be a parameter and not a fixed string
        print('Starting scan')
        scan_res = nm.scan(f'-sT {NETWORK}/24 -p {WS_PORT}')
        print('Scan ended')
        active_network_ips = scan_res['scan'].keys()
        print(f'Active network ips {scan_res["scan"]}')
        genkisockets = {ip : scan_res['scan'][f'{ip}']['tcp'][WS_PORT]['state'] \
                        for ip in active_network_ips \
                        if scan_res['scan'][f'{ip}']['tcp'][WS_PORT]['state'] == 'open'}
        print(f'result {genkisockets}')

        #Saves the result to the database


        #Returns a status 200 to the data scientist requiring the scan
        await self.send(json.dumps(genkisockets))

        #Returns something else if the process fails


