import requests as r
from io import StringIO
from Bio import SeqIO
import xml.etree.ElementTree as ET

def get_uniprot_seq(protein_id):
    print('>> Fetching UniProt Sequences for ID: ', protein_id)
    baseUrl = "http://www.uniprot.org/uniprot/"
    currentUrl = baseUrl + protein_id + ".fasta"
    response = r.post(currentUrl)
    cData = ''.join(response.text)
    Seq = StringIO(cData)
    pSeq = list(SeqIO.parse(Seq, 'fasta'))
    try:
        return str(pSeq[0].seq)
    except:
        IndexError
        return str('')


def get_isoforms(protein_id):
    print('>> Fetching UniProt Isoforms for ID: ', protein_id)
    try:
        # a dictionary storing the sequence of your isoforms, key: accesion number, value: sequence
        isoforms = dict()
        # make a call to EBI API
        req = r.get('https://www.ebi.ac.uk/proteins/api/proteins/{}/isoforms'.format(protein_id))
        # parse the returned XML
        uniprot = ET.fromstring(req.text)
        for isoform in uniprot:
            # get the sequence
            seq = isoform.find('{http://uniprot.org/uniprot}sequence')

            # get the accession number
            iso_accession = isoform.find('{http://uniprot.org/uniprot}accession')

            # add the values to the dictionary
            if seq.text and iso_accession.text:
                isoforms[iso_accession.text] = seq.text
        return isoforms
    except:
        AttributeError
        isoforms = {}
        return isoforms
