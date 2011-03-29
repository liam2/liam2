import csv
import itertools
import os
from os import path
import re
import sys

import yaml

from expr import *
from align_txt2csv import convert_txt_align

#FIXME: regressions are utterly broken... should use logit_regr(x, filter=xxx)
#instead of logit_regr(where(xxx, x, 0))

#TODO manually:        
# - if(p_yob=2003-60, MINR[2003], ...
#   ->
#   if((p_yob >= 1943) & (p_yob <= 2000), MINR[t2idx(p_yob + 60)], 0)

#TODO
# - convert "leaf" expression literals to the type of the variable being 
#   defined (only absolutely needed for bool)
# - alignement
# - remove useless bounds (eg age)
# - better automatic indentation
# - implement choose for top-level filter
# - comment out all fields unless they are actually used in one of the 
#   expressions which are in the agespine
# ? build variable dependency tree and enclose any field which is used before it
#   is computed in a lag function
# ? generic if -> choose transformation: 
#   if(c1, v1, if(c2, v2, if(c3, v3, if(c4, v4, 0))))
#   ->
#   choose(c1, v1,
#          c2, v2,
#          c3, v3,
#          c4, v4)
# ? include original comments
# ? extract common condition parts in a filter to the choose function?
# ? min/max as integer values where possible
# ? implement between

#TODO manually
# divorce function 
# KillPerson: what is not handled by normal "kill" function

def rename_var(name):
    return name

#    if name.startswith('co') and not name.startswith('coef'):
#        name = name[2:]
#        if name[0] == '_':
#            name = name[1:]
#    return name


def load_txt_def(input_path, name_idx):
    with open(input_path, "rb") as f:
        lines = list(csv.reader(f, delimiter='\t'))
    firstline = lines[0]
    colnames = firstline[:name_idx] + firstline[name_idx+1:]
    current_obj = None
    data = {}
    for line in lines[1:]:
        if not line:
            continue
        name, line = line[name_idx], line[:name_idx] + line[name_idx+1:]
        if name.startswith('first_'):
            current_obj = name[6:]
            data[current_obj] = {}
            print "reading '%s' variables" % current_obj
        elif name.startswith('end_'):
            current_obj = None
            print "done"
        elif current_obj is not None: 
            data[current_obj][name] = dict(zip(colnames, line))
    return data

def load_links(input_path):
    return load_txt_def(input_path, 0)['linkage']

def load_fields(input_path):
    data = load_txt_def(input_path, 1)
    typemap = {
        'char': float, #int, 
        'int': int,
        'int1000': float
    }
    for obj_type, obj_fields in data.iteritems():
        for name, fdef in obj_fields.iteritems(): 
            real_dtype = typemap.get(fdef['Type'], None)
            if real_dtype is None:
                print "Warning: unknown type", fdef['Type']  
            if int(fdef['nCategory']) == 2:
                assert fdef['Categories'] == "[0,1]"
                real_dtype = bool
            obj_fields[name] = {'type': real_dtype}
    return data

def load_constants(input_path):
    # macro.av is a csv with tabs OR spaces as separator and a header of 1 line
    with open(input_path, "rb") as f:
        lines = [line.replace('--', 'NaN').split()
                 for line in f.read().splitlines()]
    # sample 1955Y1 2060Y1
    firstline = lines[0]
    assert firstline[0] == "sample"
    def year_str2int(s):
        return int(s.replace('Y1', ''))
    start, stop = year_str2int(firstline[1]), year_str2int(firstline[2])
    num_periods = stop - start + 1
    names = ['YEAR'] + sorted([line[0] 
                               for line in lines[1:] 
                               if line[0] != 'YEAR'])
    data = dict((line[0], [float(cell) for cell in line[1:]])
                for line in lines[1:])
    assert len(data[names[0]]) == num_periods
    data['YEAR'] = [int(cell) for cell in data['YEAR']]
    transposed = [tuple([data[name][period] for name in names])
                  for period in range(num_periods)]
    return (start, stop), names, transposed

def load_agespine(input_path):
    # read process names until "end_spine"
    with open(input_path, "rb") as f:
        lines = [line for line in f.read().splitlines() if line]
    lines = list(itertools.takewhile(lambda l: l != 'end_spine', lines))
    #FIXME: this assumes a single obj_type for the whole list
    proc_names = [line.split('_', 2)[2] if '_p_' in line else line
                  for line in lines]
    proc_names = [rename_var(name) for name in proc_names]
    return [{'p': proc_names}]

# ================================

class TextImporter(object):
    keywords = None
    
    def __init__(self, input_path, fields):
        self.input_path = input_path
        self.fields = fields
#        print "fields", fields
        self.current_condition = None
        self.conditions = None

    def unimplemented(self, pos, line, lines):
        print "unimplemented keyword: %s" % line[0]
        return pos + 1, None
    
    def skipline(self, pos, line, lines):
        return pos + 1, None
    
    def skipifzero(self, pos, line, lines):
        if len(line) > 1 and line[1] and float(line[1]):
            print "unimplemented keyword:", line[0]
        return pos + 1, None
    
    def readvalues(self, *args):
        def f(pos, line, lines):
            values = [func(str_value)
                      for func, str_value in zip(args, line[1:])
                      if func is not None]
            if len(args) == 1:
                empty = {str: '', int: 0}
                values = values[0] if values else empty[args[0]]
            return pos + 1, values
        return f
    
    def end(self, *args):
        raise StopIteration
        
    def numorcond(self, pos, line, lines):
        # int=m, + skip line + m * (skipword, int=numand, +
        #                           numand * (str, float=min, float=max))
        num_or = int(line[1])
        pos += 2
        or_conds = []
        for i in range(num_or):
            line = lines[pos]
            num_and = int(line[1]) if len(line) >= 1 else 0
            and_conds = [(line[2 + j * 3], 
                                  float(line[3 + j * 3]), 
                                  float(line[4 + j * 3]))
                         for j in range(num_and)]
            or_conds.append(and_conds)
            pos += 1
        self.conditions[self.current_condition] = {'condition': or_conds}
        if self.current_condition == len(self.conditions) - 1:
            res = self.conditions
        else:
            res = None
        return pos, res
    
    def condition(self, pos, line, lines):
        self.current_condition = int(line[1]) - 1
        return pos + 1, None
    
    def numconditions(self, pos, line, lines):
        self.conditions = [None] * int(line[1])
        return pos + 1, None
    
    def load_txt_file(self):
        # regr_p_alive_f.txt -> alive_f:
        fpath, fname = path.split(self.input_path)
        basename, ext = path.splitext(fname)
        chunks = basename.split('_')
        assert len(chunks[1]) == 1
        name = '_'.join(chunks[2:])
    
        with open(self.input_path, "rb") as f:
            lines = list(csv.reader(f, delimiter='\t'))
    
        values = {'name': name}
        pos = 0
        while pos < len(lines):
            line = lines[pos]
            if not line:
                pos += 1
                continue
            
            keyword = line[0].lower()
            if not keyword or keyword.isspace():
                pos += 1
                continue
                
            f = self.keywords.get(keyword)
            if f is None:
                print "unknown keyword: '%s'" % keyword
                pos += 1
                continue
                
            try:
                pos, value = f(pos, line, lines)
                if value is not None:
                    values[keyword] = value
            except StopIteration:
                break
                
        return values

    # ------------------------
    # transform to expression
    # ------------------------
    
    def var_type(self, name):
        var_def = self.fields.get(name)
        if var_def is None:
            print "Warning: field '%s' not found (assuming int) !" % name
            return int
        else:
            return var_def['type']

    def var_name(self, name):
        assert name[1] == '_'
        name = name[2:]
        return rename_var(name)

    def simplecond2expr(self, cond):
        name, minvalue, maxvalue = cond
        v = Variable(self.var_name(name), self.var_type(name))
        return (v >= minvalue) & (v <= maxvalue)
        
    def andcond2expr(self, andconditions):
        assert andconditions
        expr = self.simplecond2expr(andconditions[0])
        for andcond in andconditions[1:]:
            expr = expr & self.simplecond2expr(andcond) 
        return expr
        
    def condition2expr(self, condition):
        assert condition
        expr = self.andcond2expr(condition[0])
        for orcond in condition[1:]:
            if orcond:
                expr = expr | self.andcond2expr(orcond)
        return expr

    def import_file(self):
        data = self.load_txt_file()
        predictor, expr = self.data2expr(data)
        return data['name'], predictor, expr 

    
class RegressionImporter(TextImporter):
    def __init__(self, input_path, fields):
        TextImporter.__init__(self, input_path, fields)
        
        # cfr readdyparam.cpp
        self.keywords = {
            'file description:': self.readvalues(str),
            # Time est toujours = 1 sauf dans trap_p_coeduach.txt
            'time': self.skipline, #readvalues(int),
            'align': self.readvalues(int),
            'predictor': self.readvalues(str, int, int, int),
            'numconditions': self.numconditions,
            'macro_align_multiple': self.unimplemented, #float,
            'mmkt_cond_var': self.unimplemented, #str,
            'mmkt_gender_var': self.unimplemented, #str,
            'macro_align': self.unimplemented, #float,
            'macro_align_relate': self.unimplemented, #str,
            'macro_align_type': self.unimplemented, #str,
            'ntransformations': self.unimplemented, #int + n * (str, int)
            'marrmkt': self.unimplemented, #int + n * (str, str -- which is then parsed)
            'condition': self.condition,
            'endoffile': self.end,
            'numorcond': self.numorcond,
            'indepentvar': self.indepentvar,
            'interactionterms': self.interactionterms,
            'u_varname': self.readvalues(str),
            's_u': self.skipifzero, # float, skipword, skipword, str (unused?)
            's_v': self.skipifzero, # float (unused in MIDAS?)
            'r': self.skipifzero,   # float (unused?)
            
            # ignore common junk
            'conditions': self.skipline,
            'distribution': self.skipline,
            'coefficients and structure': self.skipline,
            'errorstructure': self.skipline
        }
        
    def indepentvar(self, pos, line, lines):
        # int = m + skip line +
        #           m * (skipword, str=name, skipword, float=min,
        #                float=max, float=coef)
        #                name="constant" is a special case
        num_vars = int(line[1])
        pos += 2
        vars = []
        readvariable = self.readvalues(str, None, float, float, float)
        for i in range(num_vars):
            line = lines[pos]
            pos, values = readvariable(pos, line, lines)
            vars.append(values)
        self.conditions[self.current_condition]['vars'] = vars
        return pos, None
    
    def interactionterms(self, pos, line, lines):
        numterms = int(line[1]) if line[1] else 0
        if numterms:
            print "unimplemented keyword: interactionterms"
        return pos + 1, None

    # ------------------------
    # transform to expression
    # ------------------------
   
    def var2expr(self, var):
        name, minvalue, maxvalue, coef = var
        if name == 'constant':
            return coef
        else:
            v = Variable(self.var_name(name), self.var_type(name))
            return ZeroClip(v, minvalue, maxvalue) * coef
    
    def vars2expr(self, vars):
        assert vars
        expr = self.var2expr(vars[0])
        for var in vars[1:]:
            expr = expr + self.var2expr(var)
        return expr
    
    def data2expr(self, data):
        conditions = data['numorcond']
        assert conditions
        
        lastcond = conditions[-1]
        cond_expr = self.condition2expr(lastcond['condition'])
        expr = Where(cond_expr, self.vars2expr(lastcond['vars']), 0)
        for cond in conditions[-2::-1]:
            cond_expr = self.condition2expr(cond['condition'])
            expr = Where(cond_expr, self.vars2expr(cond['vars']), expr)
        predictor, pred_type, _, _  = data['predictor']
        assert predictor[1] == "_"
        predictor = predictor[2:]
        
        kwargs = {}
        if data.get('u_varname'):
            # another option would be to do:
            #expr += Variable(self.var_name(data['u_varname']))"
            kwargs['error_var'] = self.var_name(data['u_varname'])

        if bool(data['align']):
            kwargs['align'] = 'al_p_%s.csv' % data['name']
            if pred_type != 2:
                print "unimplemented align for pred_type:", pred_type
              
        if pred_type == 0:   # continuous
            expr = ContRegr(expr, **kwargs)
        elif pred_type == 1: # clipped continuous
            expr = ClipRegr(expr, **kwargs)
        elif pred_type == 2: # logit
            expr = LogitRegr(expr, **kwargs)
        elif pred_type == 3: # logged continuous
            expr = LogRegr(expr, **kwargs) 
        elif pred_type == 4: # clipped logged continuous
            print "Converting clipped logged continuous to logged continuous"
            expr = LogRegr(expr, **kwargs)
        else:
            print "unimplemented predictor type:", pred_type

        return predictor, expr
    
    
class TransitionImporter(TextImporter):
    def __init__(self, input_path, fields, constants, links):
        TextImporter.__init__(self, input_path, fields)
        self.constants = constants
        self.links = links
        # cfr readdyparam.cpp
        self.keywords = {
            'file description:': self.readvalues(str),
            # Time est toujours = 1 sauf dans trap_p_coeduach.txt
            'time': self.skipline, #readvalues(int),
            'align': self.readvalues(int),
            'predictor': self.readvalues(str, int),
            'numconditions': self.numconditions,
            'condition': self.condition,
            'endoffile': self.end,
            'numorcond': self.numorcond,

            'gen': self.gen, # str
            'fpbcalc': self.fpbcalc, # str
            'fgen': self.fgen, # str
            
            'zero': self.skipifzero,
            
            'first': self.skipifzero,
            'second': self.skipifzero,
            'third': self.skipifzero,

            # ignore common junk
            'conditions': self.skipline,
            'type': self.skipline,
        }
    
    def gen(self, pos, line, lines):
        # min(arg1, arg2)
        # max(arg1, arg2)
        # setto[value]
        # expression with "( ) + - * / ^ , min max"
        s = line[1]
        # add spaces around operators
        s = re.sub(r'([+\-*/^])', r' \1 ', s)
        s = s.replace('^', '**')
        s = re.sub(r'setto\[([^\]]+)\]', r'\1', s)
        self.conditions[self.current_condition]['action'] = s
        return pos + 1, None

    def fgen(self, pos, line, lines):
        # function(args)
        # - KillPerson(varname1=amount1;varnamz2=amount2;...)
        #   -> also updates marital status of spouse
        # - duration(variable,const)
        #   -> const is 1 char
        # ... (see smile p.13, 14 and 15)
        s = line[1]
        s = s.replace('CreatePerson(', "new('person', ")
        s = s.replace('newbirth(', "new('person', ")
        s = s.replace('newhousehold(', "new('household', ")
        s = re.sub(r'duration\((\w+),(\d+)\)', r'duration(\1 == \2)', s)
        # remove extra , inserted by above replacements
        s = s.replace(', )', ')')
        s = s.replace(';', ', ')
        # getlink(ps,p_inc) -> ps.p_inc
        if "getlink" in s:
            s = re.sub(r'getlink\((\w{2}),(\w+)\)', r'\1.\2', s)
            link, var = s.split('.')
            assert var[1] == '_'
            var = var[2:]
            s = "%s.%s" % (link, var)
        s = s.replace('prev(', 'lag(')
        # prevent name collision
        s = s.replace('divorce(', 'do_divorce(')
        self.conditions[self.current_condition]['action'] = s
        return pos + 1, None

    def fpbcalc(self, pos, line, lines):
        #TODO:
        # - Handle uppercase variables correctly (they come from macro.av).
        #   They exist in two forms: MINR and MINR[2003Y1].

        #TODO manually:        
        # - if(p_yob=2003-60, MINR[2003], ...
        #   ->
        #   MINR[t2idx(p_yob + 60)]
        #   note that p_yob does not seem to be computed anywhere: "yob" 
        #   does not appear anywhere in the code, nor in a transition computing
        #   it; so I guess it's not used anymore.
        s = line[1]
        
#        s = s.replace('if(', 'where(')
        s = s.replace(' and ', ' & ')
        s = s.replace(' or ', ' | ')
        # add space around operators, if not present
        s = re.sub(r'(\S)([+\-*/^])(\S)', r'\1 \2 \3', s)
        s = re.sub(r'([<>]=?)', r' \1 ', s)
        # = -> ==
        s = re.sub(r'([^<>])=', r'\1 == ', s)
        # CONST[ddddYd] -> CONST[dddd]
        s = re.sub(r'([A-Z_][A-Z0-9_]*)\[(\d{4})Y1\]', r'\1[\2]', s)
        self.conditions[self.current_condition]['action'] = s
        return pos + 1, None
    
#    def zero(self, pos, line, lines):
#        if line[1] != "0":
            # 1) find line with "predict" keyword
            # 2) for each pred_cat, cond.value[n] = float(word[n+1])
            # 3) find line with "mean" keyword
            # 4) cond.nZero = int(word[1])
            # 5) for each pred_cat, cond.mean[n] = float(word[n+1])

    # ------------------------
    # tranform to expression
    # ------------------------
   
    def action2expr(self, data):
        const_sample, const_names, const_data = self.constants
        start, stop = const_sample
        time_constants = [('Y%d' % year, Constant('Y%d' % year, year - start))
                          for year in range(start, stop + 1)]
        globals = dict(time_constants)
        globals.update((name, SubscriptableVariable(name))
                       for name in const_names)
        
        globals.update((name, Variable(self.var_name(name), 
                                       self.var_type(name)))
                       for name in self.fields.keys())
        links = [(name, Link(name, link_def['keyorig'], link_def['desttype']))
                 for name, link_def in self.links.iteritems()]
        globals.update(links)
        
        return parse(data, globals)

    def data2expr(self, data):
        # pred_type seem to be ignored for transitions
        predictor, pred_type = data['predictor']
        assert predictor[1] == "_"
        local_name = predictor[2:]
        
        conditions = data['numorcond']
        assert conditions
        
        lastcond = conditions[-1]
        cond_expr = self.condition2expr(lastcond['condition'])
        v = Variable(local_name, self.var_type(predictor))
        expr = Where(cond_expr, self.action2expr(lastcond['action']), v)
        for cond in conditions[-2::-1]:
            cond_expr = self.condition2expr(cond['condition'])
            expr = Where(cond_expr, self.action2expr(cond['action']), expr)
        return local_name, expr 


# =====================

def load_processes(input_path, input_fname, obj_type, 
                   fields, constants, links):
    #TODO: use glob instead
    if input_fname is None:
        fnames = os.listdir(input_path)
    else:
        fnames = [input_fname]
    
    data = {}    
    for fname in fnames:
        basename, ext = path.splitext(fname)
        if ext != '.txt':
#            print "skipping '%s'" % fname
            continue
        chunks = basename.split('_')
        if len(chunks) < 3: # tran_p_x
            continue
        if chunks[0] == 'al':
            continue
        if len(chunks[1]) != 1:
            continue
        if chunks[1] != obj_type:
            continue
        
        fpath = path.join(input_path, fname)
        if fname.startswith('regr_'):
            importer = RegressionImporter(fpath, fields)
        elif fname.startswith('tran_'):
            importer = TransitionImporter(fpath, fields, constants, links)
        elif fname.startswith('trap_'):
            importer = TrapImporter(fpath, fields)
        else:
            print "skipping '%s'" % fname
            importer, exporter = None, None

        if importer is not None:
            print "processing '%s'" % fname
#            name = '_'.join(chunks[2:])
            name, predictor, expr = importer.import_file()
            name, predictor = rename_var(name), rename_var(predictor)
            str_expr = str(simplify(expr))
            if name == predictor:
                res = str_expr
            else:
                # print "%s (%s)" % (name, predictor)
                res = {'predictor': predictor, 
                       'expr': str_expr}
            if name in data:
                name = name + "_duplicate"
            assert name not in data 
            data[name] = res
    return data

def convert_all_align(input_path):
    import glob
    for fpath in glob.glob(path.join(input_path, 'al_regr_*.txt')):
        convert_txt_align(fpath)

# =====================
# OUTPUT
# =====================

def orderedmap2yaml(items, indent):
    sep = '\n' + '    ' * indent
    return sep.join("- %s: %s" % f for f in items)

def links2yaml(links):
    if links:
        # ('hp', {'desttype': 'p', 'prefix': 'p',
        #         'origintype': 'h', 'keyorig': 'pid'})]
        sep = '\n            '
        return '''

        links:
            %s''' % sep.join("%s: {type: many2one, target: %s, field: %s}" %
                             (name, l['desttype'], l['keyorig'])
                             for name, l in links)
    else:
        return ''
    
def vars2yaml(processes):
    if processes:
        sep = '\n            '
        vars_str = []
        for name, expr in sorted(processes.items()):
            if isinstance(expr, dict):
                var_str = '''%s:
                predictor: %s
                expr: "%s"''' % (name, expr['predictor'], expr['expr'])
            else:
                var_str = '%s: "%s"' % (name, expr) 
            vars_str.append(var_str)
        return '''

        processes:
            %s''' % sep.join(vars_str)
    else:
        return ''

def constants2yaml(constants):
    constants = [(name, 'float') for name in constants[1]] 
    return orderedmap2yaml(constants, 2)

def entities2yaml(entities):
    entity_tmpl = "    %s:%s%s%s\n"
    e_strings = []
    for name, entity in entities.iteritems():
        fields = entity['fields']
        if fields:
            fields = sorted([(rename_var(fname), f['type'].__name__)
                             for fname, f in fields.iteritems()]) 
            fields_str = '\n        fields:\n            %s' \
                         % orderedmap2yaml(fields, 3)
        else:
            fields_str = ''
        links_str = links2yaml(entity['links'])
        vars_str = vars2yaml(entity['variables'])
        e_strings.append(entity_tmpl % (name, fields_str, links_str, vars_str))
    return '\n'.join(e_strings)

def process_list2yaml(processes):
    #[{'p': []}]
    s = []
    for d in processes:
        for entity, processes in d.items():
            p_str = ',\n              '.join(processes)
            s.append('        - %s: [%s]' % (entity, p_str))  
    return '\n'.join(s)

def simulation2yaml(constants, entities, process_list):    
    constants_str = constants2yaml(constants)
    entities_str = entities2yaml(entities)
    process_list_str = process_list2yaml(process_list)                
    return """globals:
    periodic:
        # period is implicit
        %s

entities: 
%s

simulation:
    processes:
%s
    input: 
        file: "base.h5"
    output: 
        file: "simulation.h5"
    start_period: 2003    # first simulated period
    periods: 20
""" % (constants_str, entities_str, process_list_str)
    
# =====================

if __name__ == '__main__':
    args = sys.argv
    if len(args) < 3:
        print "Usage: %s [input_path] [output_path]" % args[0]
        sys.exit()
    else:
        input_path = args[1]
        output_path = args[2]

    if not path.isdir(input_path):
        input_path, fname = path.split(input_path)
    else:
        fname = None

    fields_per_obj = load_fields(path.join(input_path, 'dyvardesc.txt'))
    constants = load_constants(path.join(input_path, 'macro.av'))
    links = load_links(path.join(input_path, 'linkage.txt'))
    process_list = load_agespine(path.join(input_path, 'agespine.txt'))
    
    fields = {}
    for obj_type, obj_fields in fields_per_obj.iteritems():
        for name, fdef in obj_fields.iteritems():
            fields['%s_%s' % (obj_type, name)] = fdef
            
    entities = {}
    for obj_type, obj_fields in fields_per_obj.iteritems():
        obj_links = [(k, v) for k, v in links.items() 
                     if v['origintype'] == obj_type]
        obj_fields.update([(v['keyorig'], {'type': int}) for k, v in obj_links])
        
        entities[obj_type] = {
            'fields': obj_fields,
            'links': obj_links,
            'variables': load_processes(input_path, fname, obj_type, 
                                        fields, constants, links)
        }

    # default YAML serialization is ugly, so we produce the string ourselves
    yamlstr = simulation2yaml(constants, entities, process_list)

    print "exporting to '%s'" % output_path
    with open(output_path, 'w') as f_out:
        f_out.write(yamlstr)
#        yaml.dump(yamldata, f_out, default_flow_style=False, 
#                  default_style='"', indent=4)
    
    if fname is None: 
        convert_all_align(input_path)
