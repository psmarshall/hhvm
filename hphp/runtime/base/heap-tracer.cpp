#include "hphp/runtime/base/heap-tracer.h"
#include "hphp/runtime/vm/jit/translator-inline.h"

namespace HPHP {

TRACE_SET_MOD(tmp2);

StaticString s_globals("GLOBALS");

void HeapTracer::traceHeap(NodeHandleFunc nodeFun) {
  VMRegAnchor _;
  const ActRec* const fp = vmfp();
  const TypedValue *const sp = (TypedValue *)vmsp();
  int offset = (fp->m_func->unit() != nullptr)
               ? fp->unit()->offsetOf(vmpc())
               : 0;

  //Main source of roots
  traceStackFrame(fp, offset, sp);

  //Also use $GLOBALS
  auto globals = g_context->m_globalVarEnv->lookup(s_globals.get());
  if (globals != nullptr) {
    FTRACE(2, "Found globals\n");
    TypedValue nulltv;
    nulltv.m_type = DataType::KindOfUninit;
    SearchNode n = {*globals, nulltv};
    m_searchQ.push(n);
  }

  FTRACE(2, "Found {} roots\n", m_searchQ.size());
  // Checks

  TypedValue cur;
  SearchNode n;

  while (!m_searchQ.empty()) {
    n = m_searchQ.front();
    cur = n.current;
    m_searchQ.pop();

    if (cur.m_type <= KindOfDouble) {
      continue; //Ignore primitive values.
    }

    if(isReachable(cur.m_data.pstr)) {
      continue;
    }

    markReachable(cur.m_data.pstr);

    //Handle this node
    nodeFun(n);

    //Find children, ma
    if(cur.m_type == DataType::KindOfArray) {
      TRACE(3, "Found array\n");
      if(cur.m_data.parr) {
        for(ArrayIter iter(cur.m_data.parr); iter; ++iter) {
          m_searchQ.push({*iter.first().asTypedValue(), cur});
          m_searchQ.push({*iter.second().asTypedValue(), cur});
        }
      } else {
        TRACE(3, "It was a fake\n");
      }
    }

    if(cur.m_type == DataType::KindOfRef) {
      TRACE(3, "Found a ref\n");
      m_searchQ.push({*cur.m_data.pref->tv(), cur});
    }

    if(cur.m_type == DataType::KindOfObject) {
      TRACE(3, "Found an object\n");
      ObjectData *od = cur.m_data.pobj;
      if(!od) {
        TRACE(3, "It was a fake\n");
        continue;
      }

      auto const cls = od->getVMClass();
      const TypedValue *props = od->propVec();
      auto propCount = cls->numDeclProperties();

      //First go through all of the properties which are defined by the class
      for(auto i = 0; i < propCount; i++) {
        const TypedValue *prop = props + i;
        m_searchQ.push({*prop, cur});
      }

      //Then, if there are dynamic properties
      if (UNLIKELY(od->getAttribute(ObjectData::Attribute::HasDynPropArr))) {
        //The dynamic properties are implemented internally as an
        //array, so why not treat them in the same way?
        auto dynProps = od->dynPropArray();

        TypedValue dynTv;
        dynTv.m_data.parr = dynProps.get();
        dynTv.m_type = DataType::KindOfArray;
        m_searchQ.push({dynTv, cur});
      }
    }
  }

  TRACE(1, "Finished tracing\n");
  reset();
}


void HeapTracer::traceStackFrame(const ActRec *fp, int offset, const TypedValue *ftop) {
  const Func *func = fp->m_func;
  func->validate();
  std::string funcName(func->fullName()->data());

  TypedValue nulltv = {};
  nulltv.m_type = DataType::KindOfUninit;

  FTRACE(2, "Visiting: {} at offset {}\n", funcName.c_str(), fp->m_soff);


  if(fp->hasThis()) {
    TypedValue thisTv;
    thisTv.m_data.pobj = fp->getThis();
    thisTv.m_type = KindOfObject;
    m_searchQ.push({thisTv, nulltv});
  }

  const int numExtra = fp->numArgs() - fp->m_func->numNonVariadicParams();
  if (numExtra > 0) {
    for (int i = 0; i < numExtra; i++) {
      TypedValue *val = fp->getExtraArgs()->getExtraArg(i);
      if (val != nullptr) {
        m_searchQ.push({*val, nulltv});
        FTRACE(1, "Extra arg: {}\n", tname(val->m_type));
      }
    }
  }

  if (fp->hasVarEnv()) {
    auto table = fp->getVarEnv()->getTable();

    for (auto it = NameValueTable::Iterator(table);
         it.valid();
         it.next()) {
      auto tv = it.curVal();
      FTRACE(1, "NVT elem: {} - {}\n", *it.curKey(), tname(tv->m_type));
      m_searchQ.push({*tv, nulltv});
    }
  }

  //Check out the local variables
  TypedValue *tv = (TypedValue *)fp;
  tv --;
  if (func->numLocals() > 0) {
    int n = func->numLocals();
    for (int i = 0; i < n; i++) {
      m_searchQ.push({*tv, nulltv});
      tv --;
    }
  }

  //See if the stack holds any secrets
  visitStackElems(fp, ftop, offset,
      [&](const ActRec *ar) {
        std::string funcName(ar->m_func->fullName()->data());
        FTRACE(1, "Found an ActRec in the stack: {}, {} at {}\n", ar, funcName.c_str(), ar->m_soff);
      },
      [&](const TypedValue *tv) {
        FTRACE(2, "Found a TypedValue in the stack: {}\n", tv);
        m_searchQ.push({*tv, nulltv});
      }
  );

  //Try and step down within the stack
  {
    Offset prevPc = 0;
    TypedValue *prevStackTop = nullptr;
    ActRec *prevFp = g_context->getPrevVMStateUNSAFE(fp, &prevPc, &prevStackTop);
    if (prevFp != nullptr) {
      traceStackFrame(prevFp, prevPc, prevStackTop);
    }
  }
}

void HeapTracer::reset() {
  marked.clear();
}

void HeapTracer::markReachable(void *ptr) {
  marked.insert(ptr);
}

bool HeapTracer::isReachable(void *ptr) {
  return (marked.find(ptr) != marked.end());
}

}
