# encoding: utf-8
from __future__ import absolute_import, division, print_function

#       Copyright (C) 2005-2007 Carabos Coop. V. All rights reserved
#       Copyright (C) 2008-2013 Vicent Mas. All rights reserved
#
#       This program is free software: you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation, either version 3 of the License, or
#       (at your option) any later version.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#
#       You should have received a copy of the GNU General Public License
#       along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#       Author:  Vicent Mas - vmas@vitables.org

import locale
import warnings
import sys

from liam2 import config
from liam2.compat import PY2
from liam2.utils import ExceptionOnGetAttr

try:
    from qtpy import QtWidgets
    from vitables.vtapp import VTApp
except ImportError as e:
    msg = "the 'view' command is not available because 'vitables' does not seem to be installed correctly (%s)." % e
    print("Warning:", msg)
    if not config.debug and not PY2:
        e = ImportError(msg).with_traceback(sys.exc_info()[2])
    VTApp = ExceptionOnGetAttr(e)
    QtWidgets = ExceptionOnGetAttr(e)


def viewhdf(filepaths):
    app = QtWidgets.QApplication(filepaths)

    # These imports must be done after the QApplication has been instantiated
    with warnings.catch_warnings():
        # ignore deprecation warnings just for this import
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from vitables.vtapp import VTApp

    from vitables.preferences import vtconfig

    # Specify the organization's Internet domain. When the Internet
    # domain is set, it is used on Mac OS X instead of the organization
    # name, since Mac OS X applications conventionally use Internet
    # domains to identify themselves
    app.setOrganizationDomain('vitables.org')
    app.setOrganizationName('ViTables')
    app.setApplicationName('ViTables')
    app.setApplicationVersion(vtconfig.getVersion())

    # Localize the application using the system locale
    # numpy seems to have problems with decimal separator in some locales
    # (catalan, german...) so C locale is always used for numbers.
    locale.setlocale(locale.LC_ALL, '')
    locale.setlocale(locale.LC_NUMERIC, 'C')

    # Start the application
    vtapp = VTApp(mode='a', h5files=filepaths)
    vtapp.gui.show()
    app.exec_()
