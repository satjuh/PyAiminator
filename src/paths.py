import json
import os

#TODO tests
class DataPath:

    def __init__(self):
        """
        """
        super().__setattr__('__dict__', {})
        self.base = ['images','video','root_data']

        save = False
        try:
            with open('paths.json', 'r') as f:
                js = json.load(f)
                base = self.check_valid(js)
                if js != base:
                    save = True
                self.default = DefaultPath(base)
        except FileNotFoundError:
            self.default = DefaultPath()
            base = self.default.as_dict()
            save = True

        if save:
            with open('paths.json', 'w') as f:
                json.dump(base, f)

        self.create_if_not_exist(base)
        self.default.sub_paths()
        self.create_if_not_exist(self.default.as_dict())

        for key in self.default.__dict__:
            self.__dict__[key] = self.default.__dict__[key]

    def __getattr__(self, key):
        """
        Gets class attribute
        Raises AttributeError if key is invalid
        """
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise AttributeError

    def __setattr__(self, key, value):
        """
        Sets class attribute according to value
        If key was not found, new attribute is added
        """
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            super().__setattr__(key, value)

    def check_valid(self, paths):

        d = {}
        for key in paths:
            if key in self.base:
                d[key] = paths[key]

        return d

    def create_if_not_exist(self, paths):

        for key in paths:
            try:
                if not os.path.exists(paths[key]):
                    os.mkdir(paths[key])
            except OSError:
                paths[key] = self.default.__dict__[key]
                if paths[key]:
                    if not os.path.exists(paths[key]):
                        os.mkdir(paths[key])
        return paths

    def get_index(self, key, name='.'):
        index = 0
        if os.listdir(self.__dict__[key]):
            indexes = sorted([int(f.split(name)[0]) for f in os.listdir(self.__dict__[key])], reverse=True)
            if indexes:
                index = indexes[0] + 1
        return index


class DefaultPath:

    def __init__(self, initial={}):
        super().__setattr__('__dict__', {})

        if not initial:
            self.__dict__['images'] = 'images'
            self.__dict__['video'] = 'video'
            self.__dict__['root_data'] = ''
        else:
            for key in initial:
                self.__dict__[key] = initial[key]

    def sub_paths(self):
        sub_dirs = [
            ('data', 'root_data'),
            ('collected', 'data'),
            ('dataframes', 'data'),
            ('combined', 'data'),
            ('templates', 'images'),
            ('processed', 'video')
        ]
        for sub_path in sub_dirs:
            self.__dict__[sub_path[0]] = os.path.join(self.__dict__[sub_path[1]], sub_path[0])

    def __getattr__(self, key):
        """
        Gets class attribute
        Raises AttributeError if key is invalid
        """
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise AttributeError

    def __setattr__(self, key, value):
        """
        Sets class attribute according to value
        If key was not found, new attribute is added
        """
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            super().__setattr__(key, value)

    def as_dict(self):
        return self.__dict__
