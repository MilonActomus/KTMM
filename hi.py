class Indenter:
    def __init__(self):
        self.count=0
    def __enter__(self):
        self.count+=1
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.count-=1

    def print(self,text):
        print('\t'*self.count + text)


with Indenter() as indent:
    indent.print('hi!')
    with indent:
        indent.print('hello')
        with indent:
            indent.print('bonjour')
    indent.print('hey')
