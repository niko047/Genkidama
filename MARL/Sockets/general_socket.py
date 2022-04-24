class GeneralSocket(object):

    def __init__(self, address, port, cores_orchestrator):
        self.address = address
        self.port = port
        self.cores_orchestrator = cores_orchestrator()

    @staticmethod
    def wait_msg_received(len_true_msg, gsocket):
        """Wait to receive the entirety of the message and accumulates in in a bytes obj"""
        # Initializes empty bytes object
        msg = b''

        while len(msg) < len_true_msg:
            # Keep stacking up bytes information
            msg += gsocket.recv(len_true_msg)

        return msg