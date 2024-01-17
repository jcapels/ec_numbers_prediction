import pickle

from lxml import etree
from tqdm import tqdm

import gc


class UniRefXmlParser:
    def __init__(self, filename):
        self.filename = filename
    
    def parse(self, filename_output_path):
        import gzip
        self.file = gzip.open(self.filename, 'rb')
        context = etree.iterparse(self.file, events=("start", "end"))

        representatives = []
        for event, elem in context:
            if (event == "start" or event == "end") and elem.tag == "{http://uniprot.org/uniref}representativeMember":
                properties = elem.xpath('./*/*[@type="UniProtKB accession"]')
                if properties is not None and len(properties) > 0:
                    prop = properties[0]
                    representatives.append(prop.attrib["value"])

            # elif event == "start" and elem.tag == "{http://uniprot.org/uniref}member" and \
            #         current_representative is not None:
            #     properties = elem.xpath('./*/*[@type="UniProtKB accession"]')
            #     # found = False
            #     if properties is not None and len(properties) > 0:
            #         prop = properties[0]
            #         if with_cluster_accessions:
            #             cluster_accessions[current_representative].append(prop.attrib["value"])
            #         if with_uniprot_accession_to_cluster:
            #             uniprot_accession_to_cluster[prop.attrib["value"]] = current_representative

            # if event == "end" and elem.tag == "{http://uniprot.org/uniref}entry":
            #     current_representative = None
            #     bar.update(1)

            #     if bar.n != 0 and bar.n % division == 0:
            #         if with_cluster_accessions:
            #             self.save(f"cluster_accessions_{counter}", cluster_accessions)
            #         if with_uniprot_accession_to_cluster:
            #             self.save(f"uniprot_accession_to_cluster_{counter}", uniprot_accession_to_cluster)
            #         counter += 1

            #         cluster_accessions.clear()
            #         uniprot_accession_to_cluster.clear()

            #         gc.collect()

            #         cluster_accessions = {}
            #         uniprot_accession_to_cluster = {}

            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

        file = open(f'{filename_output_path}', 'wb')
        pickle.dump(representatives, file)
        file.close()

