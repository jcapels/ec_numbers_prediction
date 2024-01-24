import pickle

from lxml import etree


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

            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

        file = open(f'{filename_output_path}', 'wb')
        pickle.dump(representatives, file)
        file.close()
