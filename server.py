#!/usr/bin/env python3

from featurizer import *
import joblib
from pysmt.smtlib.printers import SmtPrinter, SmtDagPrinter
from pysmt.smtlib.parser import SmtLibParser,SmtLibExecutionCache

model = joblib.load('models/lgb.pkl')

parser = SmtLibParser(interactive=True)
import socketserver

class MyTCPHandler(socketserver.BaseRequestHandler):
    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """
    def handle(self):
        # self.request is the TCP socket connected to the client
        global parser
        global dec
        global env
        BUFF_SIZE = 1024
        data = b""
        while True:
            part = self.request.recv(BUFF_SIZE)
            data += part
            if len(part) < BUFF_SIZE:
                break
        data = data.decode()
        ix = data.index("#")
        message_type = data[ix+1] # a = declaration, b = quantifier, E=shutdown
        data =  data[ix+2:]
        if message_type=='a':
            #with open('data/data/declarations.log', 'r') as f:
            #    data = f.read()
            dec = data.split("\n")
            pysmt.environment.reset_env()
            parser = SmtLibParser(interactive=True)
            parsed = parser.get_script(get_dec(dec))
            self.request.sendall(b'2#ok')
            return
        elif message_type=='b':
            quantifier = data
            #with open('data/data/quantifier.txt', 'r') as f:
            #    quantifier = f.read()
        elif message_type=='c':
            #with open('data/data/dec2.txt', 'r') as f:
            #    data = f.read()
            dec = data.split("\n")
            parsed = get_script(parser, get_dec(dec))
            self.request.sendall(b'2#ok')
            return
        elif message_type=='E':
            self.server._BaseServer__shutdown_request = True
            self.request.sendall(b'7#exiting')
            return
        extracted_data_per_formula,var_term_counts = get_parsed_format(parser,quantifier)
        split_ixs = []
        ix = 0
        for i in var_term_counts[:-1]:
            ix += i
            split_ixs.append(ix)

        feature_vectors = get_feature_vectors(extracted_data_per_formula)
        output = model.predict(feature_vectors)
        output_split = [list(i) for i in np.split(output,split_ixs)]
        output_b = str.encode(str(output_split).replace(",",""))
        out_data =  bytes(str(len(output_b)),'utf-8') + b'#'+ output_b
        self.request.sendall(out_data)

if __name__ == "__main__":
    HOST, PORT = "127.0.0.1", 8080

    with socketserver.TCPServer((HOST, PORT), MyTCPHandler) as server:
        # Activate the server; this will keep running until you
        # interrupt the program with Ctrl-C
        server.serve_forever()
