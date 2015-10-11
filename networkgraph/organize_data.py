#!/usr/bin/env python
import sys
import json
from copy import *
#import re
#import codecs
import logging
import argparse

def network_args():
    """
    Returns a parser that can be used to view a list of the arguments.
    """
    parser = argparse.ArgumentParser(
            description="Build nodes and links from two columns of text set up as userID followerID.  Each line is a relationship.")
    parser.add_argument('-i'
            , '--infile'
            , dest="infile"
            #, nargs='?'
            , type=argparse.FileType('r')
            , default=sys.stdin
            , help="Input file name (optional, if omitted expects input from stdin).")
    parser.add_argument("-d"
            , "--delimiter"
            , dest="delimiter"
            , default=","
            , help="Delimator type name (default = ',').")
    parser.add_argument('-l'
            , '--labels'
            , dest="labels"
            #, nargs='?'
            , type=argparse.FileType('r')
            , default=None
            , help="File path to label the dataset with a csv in the format userID,label.")
    parser.add_argument("-m"
            , "--minimum number of links"
            , dest="minLinks"
            , default=0
            , help="Nodes w/ link counts great than or equal to this value are kept in the graph.")
    parser.add_argument('-o'
            , dest="outfile"
            #'--infile'
            #, nargs='?'
            , type=argparse.FileType('w')
            , default=sys.stdout
            , help="Output file name (optional, if omitted expects input from stdin).")
    return parser

def get_labels(dataFile,delim):
    d = {}
    for i,line in enumerate(dataFile):
        # skip header
        if i==1:
            continue
        rec = line.strip().split(delim)
        userID = rec[1]
        cluster = rec[2]
        d[userID]=cluster
    return d

class Graph(object):
    """
    Methods create the nodes and links for a network graph.
    """
    def __init__(self
            , nodes=[]
            , links = []
            , infile = None
            , delimiter = ','
            , minLinks = 0
            ):
        self.nodes = nodes
        self.links = links
        self.infile = infile
        self.delimiter = delimiter
        self.minLinks = int(minLinks)

    def increment_index(self,index,indexCounter,ID):
        index[ID]=indexCounter
        indexCounter+=1
        return index,indexCounter

    def append_node(self,ID,group):
        self.nodes.append({"name":str(ID),"group":group})

    def append_link(self,index, userID,followerID):
        self.links.append({"source":index[userID],"target":index[followerID]})

    def build_graph(self,data):
        """
        Builds an index for the values in the node list and fills the target list.
        """
        index = {}
        indexCounter = 0
        for line in self.infile:
            # parse file
            ids = line.strip().split(self.delimiter)
            userID = ids[1]
            followerID = ids[0]
            # add userID to index 
            if userID not in index:
                # increment the indexCounter and add userID to index
                index,indexCounter=self.increment_index(index,indexCounter,userID)
                group = 0
                # give userID the appropriate group information
                #sys.stderr.write("{},{}\n".format(type(userID),userID))
                if userID in data:
                    group = int(data[userID])
                    #sys.stdout.write("{},{}\n".format(userID,group))
                self.append_node(userID,group)
            # add followerID to index 
            if followerID not in index:
                index,indexCounter=self.increment_index(index,indexCounter,followerID)
                group = 1
                if followerID in data:
                    group = int(data[followerID])
                    #sys.stdout.write("{},{}\n".format(userID,group))
                self.append_node(followerID,group)
            self.append_link(index,userID,followerID)

    def remove_nodes(self):
        """
        Keeps nodes that meet the minimum number of links.
        """
        n = self.minLinks
        linkCount = {}
        keepers = []
        index = {}
        g.newLinks=[]
        # count the number of links per source node
        for node in self.links:
            linkCount[node['source']] = linkCount.get(node['source'],0)+1
        # generate a list of nodes to keep
        for index,count in linkCount.iteritems():
            if count >= n:
                keepers.append(index)
        # add to this list each of those nodes connections
        for item in self.links:
            if ((item['source'] in keepers) or (item['target'] in keepers)):
                if item['source'] not in keepers:
                    keepers.append(item['source'])
                if item['target'] not in keepers:
                    keepers.append(item['target'])
        # generate a mapping of old_index:new_index
        mapping = {v:k for k,v in dict(enumerate(keepers)).items()}
        # remove nodes that do not meet the minimum nubmer of links
        self.nodes = [self.nodes[i] for i in keepers]
        # remove unnecessary links by reindexing links w/ the new index of the nodes
            #self.newlinks = [{"source":mapping[item['source']], "target":mapping[item['target']]} for item in self.links if (item['source'] in keepers)]
        for item in self.links:
            if (item['source'] in keepers):
                try:
                    g.newLinks.append({"source":mapping[item['source']], "target":mapping[item['target']]})
                except KeyError,e:
                    sys.stderr.write(str(e)+'\n')
        with open('users.csv','wb') as users:
            for node in self.nodes:
                users.write(node['name']+'\n')



if __name__ == '__main__':
    # commandline options in a namespace (dict)
    options = network_args().parse_args()
    # get data labels
    clusterID = {}
    if options.labels is not None:
        clusterID = get_labels(options.labels,options.delimiter)
    # create a Graph instance
    g = Graph(infile=options.infile, delimiter=options.delimiter, minLinks = options.minLinks)
    # build the nodes and links
    g.build_graph(clusterID)
    # remove nodes with few links (not currently in use)
    #if g.minLinks > 0:
    #    g.remove_nodes()
    # write the output
    graph = {"nodes":g.nodes,"links":g.links}
    options.outfile.write(json.dumps(graph))
