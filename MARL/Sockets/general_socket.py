class GeneralSocket(object):

    def __init__(self, address, port, neural_net):
        self.address = address
        self.port = port
        self.neural_net = neural_net()

    @staticmethod
    def wait_msg_received(len_true_msg, len_arriving_msg, gsocket):
        """Wait to receive the entirety of the message and accumulates in in a bytes obj"""
        # Initializes empty bytes object
        msg = b''

        while len_arriving_msg < len_true_msg:
            # Keep stacking up bytes information
            msg += gsocket.recv(len_true_msg)

        return msg