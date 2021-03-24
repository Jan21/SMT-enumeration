
from featurizer import *
import joblib

model = joblib.load('models/lgb.pkl')



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
        BUFF_SIZE = 1024
        data = b""
        
        while True:
            part = self.request.recv(BUFF_SIZE)
            data += part
            if len(part) < BUFF_SIZE:
                break
        ix = data.index(b"#")
        data =  data[ix+1:]
        #message = json.loads(data.decode())

        #print(f"Received {message!r}")

        with open('data/declarations.log','r') as f:
            log = f.readlines()

        with open('data/quantifier.txt','r') as f:
            quantifier = f.read()
        
        extracted_data_per_formula,var_term_counts = get_parsed_format(quantifier,log)
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
