import subprocess
import luigi


def runcmd(cmd, verbose=False, *args, **kwargs):
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass


class DownloadSwissProt(luigi.Task):

    def requires(self):
        return []

    def output(self):
        return luigi.LocalTarget('uniprot_sprot.xml.gz')

    def run(self):
        url = 'http://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.xml.gz'
        runcmd(f'wget {url}')


class DownloadTrembl(luigi.Task):

    def requires(self):
        return []

    def output(self):
        return luigi.LocalTarget('uniprot_trembl.xml.gz')

    def run(self):
        url = 'http://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.xml.gz'
        runcmd(f'wget {url}')


class DownloadUniref(luigi.Task):

    def requires(self):
        return []

    def output(self):
        return luigi.LocalTarget('uniref90.xml.gz')

    def run(self):
        url = 'http://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.xml.gz'
        runcmd(f'wget {url}')
