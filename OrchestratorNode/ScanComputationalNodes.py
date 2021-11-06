'''import json
from channels.generic.websocket import WebsocketConsumer
from NodeLog.models import NodeLog
import nmap
import time

class ScanComputationalNodes(WebsocketConsumer):

    # groups = ["broadcast"]

    def connect(self):
        #Accepts only requests from itself (or very trusted nodes)
        self.accept()

    def receive(self, text_data=None, bytes_data=None):

        WS_PORT = 7999
        NETWORK = '192.168.1.0'

        #Pings the networks and checks for pingable IPs
        nm = nmap.PortScanner()

        print('Starting scan')
        start_time = time.time()

        scan_res = nm.scan(f'-sS {NETWORK}/24 -p {WS_PORT}')

        end_time = time.time()
        print(f'Scan ended, time elapsed: {round(end_time - start_time)} seconds')

        active_network_ips = scan_res['scan'].keys()
        print(f'Active network ips {scan_res["scan"]}')
        genkisockets = {ip : scan_res['scan'][f'{ip}']['tcp'][WS_PORT]['state'] \
                        for ip in active_network_ips \
                        if scan_res['scan'][f'{ip}']['tcp'][WS_PORT]['state'] == 'open'}
        print(f'result {genkisockets}')

        #Returns a status 200 to the data scientist requiring the scan
        self.send(json.dumps(genkisockets))

        # Saves the result to the database


        #Returns something else if the process fails
'''

