import Image as im
import numpy as np
import queue

class Huffman(object):

    def __init__(self, image):
        self.image = image
        self.pathSave = image.path.getPathSave(image.name)
        self.extension_compress = "_compressed.bin"
        self.histogram = im.Histogram().getValues(self.image.arr)

        self.m, self.n = image.shapes
        self.root_node = None
        self.input_binary = None
        self.bitstream = None
        self.count = 0

    def compress(self):
        self.bitstream = ['0'] * len(self.histogram)
        probabilities = self.histogram/np.sum(self.histogram)

        self.root_node = self.tree(probabilities)
        self.makeTrail(self.root_node, np.ones([64], dtype=int))
        
        self.saveDictionary(self.bitstream)
        self.saveBytes(self.bitstream)

        start_bits = self.m * self.n * 8
        end_bits = len(self.input_binary)

        compression = end_bits/start_bits
        redundancy = 1 - (1/(start_bits/end_bits))

        print('Grau de compressão:', compression)
        print('Redundância relativa:', redundancy)
        
    def decompress(self):
        binary = self.readBytes()

        print(self.root_node)
        print(len(binary))


    def readBytes(self):
        input_path = self.pathSave + self.extension_compress

        with open(input_path, 'rb') as file:
            bit_string = ""
            byte = file.read(1)

            while(len(byte) > 0):
                byte = ord(byte)
                bits = bin(byte)[2:].rjust(8, '0')
                bit_string += bits
                byte = file.read(1)

            encoded_text = self.removePadding(bit_string)
            return encoded_text

    def saveBytes(self, bitstream):
        output_path = (self.pathSave + self.extension_compress)
        b = bytearray()
        binary = ''

        for y in range(self.m):
            for x in range(self.n):
                binary += bitstream[self.image.arr[y,x]]

        self.input_binary = binary
        binary = self.addPadding(binary)

        with open(output_path, 'wb') as output:
            for i in range(0, len(binary), 8):
                b.append(int(binary[i:i+8], 2))
            output.write(bytes(b))

    def addPadding(self, binary):
        extra_padding = 8 - len(binary) % 8
        for _ in range(extra_padding):
            binary += "0"
        padded_info = "{0:08b}".format(extra_padding)
        binary = padded_info + binary
        return binary

    def removePadding(self, binary):
        padded_info = binary[:8]
        extra_padding = int(padded_info, 2)
        binary = binary[8:] 
        encoded_text = binary[:-1*extra_padding]
        return encoded_text

    def saveDictionary(self, bitstream):
        f = open((self.pathSave + "_dictionary.txt"),'w')
        for (index, element) in enumerate(bitstream):
            f.write(str(index) + ' ' + element + '\n')

    def makeTrail(self, root_node, tmp_array):
        if (root_node.left is not None):
            tmp_array[self.count] = 0
            self.count+=1
            self.makeTrail(root_node.left, tmp_array)
            self.count-=1
        if (root_node.right is not None):
            tmp_array[self.count] = 1
            self.count+=1
            self.makeTrail(root_node.right, tmp_array)
            self.count-=1
        else:
            binary = ''.join(str(cell) for cell in tmp_array[1:self.count])
            self.bitstream[root_node.data] = binary

    def tree(self, probabilities):
        prq = queue.PriorityQueue()
        for (color, probability) in enumerate(probabilities): 
            leaf = Node() 
            leaf.data = color
            leaf.prob = probability
            prq.put(leaf)
        
        while (prq.qsize() > 1):
            newnode = Node() 
            l = prq.get() 
            r = prq.get()
            newnode.left = l 
            newnode.right = r 
            newprob = l.prob + r.prob
            newnode.prob = newprob
            prq.put(newnode) 
        return prq.get()

class Node:
	def __init__(self):
		self.prob = None
		self.code = None
		self.data = None
		self.left = None
		self.right = None

	def __lt__(self, other):
		return 1 if (self.prob < other.prob) else 0

	def __ge__(self, other):
		return 1 if (self.prob > other.prob) else 0