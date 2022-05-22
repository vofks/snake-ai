import csv
import os
import datetime

RESULT_FOLDER = 'results'
FILE_FORMAT = '.csv'


class ExperimentLog:
    def __init__(self, project_name, timestamp):
        self.filename = project_name + '_' + \
            timestamp.strftime('%H-%M-%S %d-%m-%Y') + FILE_FORMAT

    def setup(self):
        if not os.path.exists(RESULT_FOLDER):
            os.mkdir(RESULT_FOLDER)

        self._path = os.path.join(RESULT_FOLDER, self.filename)

        if os.path.exists(self._path):
            print(
                f'Project log error. File: {self._path} already exists. Abort.')
            return False

        self._new = True

        return True

    def logrow(self, **kwargs):
        with open(self._path, 'a', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=kwargs.keys(),  quoting=csv.QUOTE_MINIMAL)

            if self._new:
                writer.writeheader()
                self._new = False

            writer.writerow(kwargs)


if __name__ == '__main__':
    log = ExperimentLog('test', datetime.datetime.now())
    success = log.setup()

    if not success:
        print('Error')

    hello = 'world'
    foo = 1234

    log.logrow(hello=hello, foo=foo, stamp=datetime.datetime.now())
    log.logrow(hello=hello, foo=foo, stamp=datetime.datetime.now())
    log.logrow(hello=hello, foo=foo, stamp=datetime.datetime.now())
