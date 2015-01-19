/*
   +----------------------------------------------------------------------+
   | HipHop for PHP                                                       |
   +----------------------------------------------------------------------+
   | Copyright (c) 2010-2014 Facebook, Inc. (http://www.facebook.com)     |
   +----------------------------------------------------------------------+
   | This source file is subject to version 3.01 of the PHP license,      |
   | that is bundled with this package in the file LICENSE, and is        |
   | available through the world-wide-web at the following url:           |
   | http://www.php.net/license/3_01.txt                                  |
   | If you did not receive a copy of the PHP license and are unable to   |
   | obtain it through the world-wide-web, please send a note to          |
   | license@php.net so we can mail you a copy immediately.               |
   +----------------------------------------------------------------------+
*/

#include <iostream>
#include <string>

#include "hphp/runtime/base/tracing-collector-impl.h"

#include "hphp/runtime/base/variable-serializer.h"
#include "hphp/runtime/base/thread-info.h"
#include "hphp/runtime/vm/jit/translator-inline.h"

#include "hphp/util/trace.h"


namespace HPHP {

///////////////////////////////////////////////////////////////////////////////

TRACE_SET_MOD(tmp1);

// External API (tracing-collector.h)

int64_t tracingGCCollect() {
  return gc().collect();
}

void markDestructableObject(ObjectData const *obj) {
  gc().markDestructable(obj);
}

void requestGC() {
  ThreadInfo *info = ThreadInfo::s_threadInfo.getNoCheck();
  info->m_reqInjectionData.setGarbageCollectionFlag();
}

void markObjectLive(void *ptr, DataType t) {
  gc().markObjectLive(ptr, t);
}


// stack trace helper
static AllocStackTrace getStackTrace() {
  AllocStackTrace trace;

  if (g_context.isNull()) return trace;
  VMRegAnchor _;
  ActRec *fp = vmfp();
  if (!fp) return trace;
  PC pc = vmpc();

  const Func *f = fp->m_func;
  Unit *u = f->unit();
  Offset off = pc - u->entry();
  for (;;) {
    trace.push_back({ f, off, fp->resumed() });
    fp = g_context->getPrevVMStateUNSAFE(fp, &off);
    if (!fp) break;
    f = fp->m_func;
  }
  return trace;
}



//Thread local singleton stuff

MarkSweepCollector::TlsWrapper tls;

MarkSweepCollector &gc() {
  return *MarkSweepCollector::TlsWrapper::getCheck();
}

void MarkSweepCollector::Create(void *storage) {
  new (storage) MarkSweepCollector();
}

void MarkSweepCollector::Delete(MarkSweepCollector *msc) {
  msc->~MarkSweepCollector();
}

void MarkSweepCollector::OnThreadExit(MarkSweepCollector *msc) {}


//Actual implementation

int64_t MarkSweepCollector::collect() {

  prepareSlabData();

  for(auto thing : slabLookup) {
    FTRACE(3, "{} -> {}\n", (void *)thing.first, (void *)m_slabs[thing.second].slab.base);
  }

  tracer.traceHeap([&] (const SearchNode n) {
      markReachable(n.current.m_data.pstr);

      //DEBUG: Insert into the trace set
      traceSet.insert(std::make_pair((void *)n.current.m_data.pstr, n.current.m_type));
      });

  int reclaimed = 0;

  for(auto sd : m_slabs) {
    FTRACE(1, "Used: {}\n{}\n\n", (void *)sd.slab.base, sd.usedBlocks.to_string());
    //FTRACE(1, "Allocated: {}\n{}\n\n", (void *)sd.slab.base, sd.slab.allocatedBlocks.to_string());

    auto blocksToFree = ~ sd.usedBlocks & sd.slab.allocatedBlocks;

    //FTRACE(1, "To free:\n{}\n\n", blocksToFree.to_string());

    //Should remain disabled until we know that gc_enabled objects are not being collected elsewhere
    //(Otherwise they could end up on a freelist as well as in the block allocator)
    //MM().recycleMemory(sd.slab, blocksToFree);
    reclaimed += blocksToFree.count();
  }

  FTRACE(1, "Reclaimed {} blocks ({} kb)\n", reclaimed, reclaimed * 4);

  checkTraceSet();

  //Clear out the intermediate data we used
  cleanData();

  return 0;
}

void MarkSweepCollector::checkTraceSet() {
  FTRACE(1, "LiveSet has {} objs, traceSet has {}\n", liveSet.size(), traceSet.size());
  for (auto elem : liveSet) {
    StringData *sd = (StringData *)elem.first;
    if (elem.first != (void *)0x7a7a7a7a7a7a7a7a && sd->getCount() > 0) {
      // For every LIVE element in the liveset,
      // check that it was found be the trace. If it was not,
      // print an error.
      //
      // We need to be careful memory isn't reclaimed when the m_count
      // hits 0 (and indeed, that the m_count actually is decremented)
      if(traceSet.find(elem) == traceSet.end()) {
        FTRACE(1, "MISSING FROM TRACE: {} at {}\n", tname(elem.second), elem.first);
        auto tr = allocTraces[elem.first];
        for (auto srckey : tr) {
          FTRACE(2, "   {}\n", show(srckey));
        }
      }
    }
  }
}

void MarkSweepCollector::printStack() {
  VMRegAnchor _;
  const ActRec* const fp = vmfp();
  int offset = (fp->m_func->unit() != nullptr)
               ? fp->unit()->offsetOf(vmpc())
               : 0;
  TRACE(1, g_context->getStack().toString(fp, offset));
}


void MarkSweepCollector::prepareSlabData() {
  const std::vector<MemoryManager::Slab> activeSlabs = MM().getActiveSlabs(true);
  const auto alignmentFactor = MemoryManager::kSlabSize / MemoryManager::kSlabAlignment;

  for(auto slab : activeSlabs) {
    auto i = m_slabs.size();
    MarkSweepCollector::SlabData sd;
    sd.slab = slab;
    m_slabs.push_back(sd);

    slabLookup[sd.slab.base] = i;
  }
}

void MarkSweepCollector::cleanData() {
  m_slabs.clear();
  slabLookup.clear();

  liveSet = traceSet;
  traceSet.clear();

  allocTraces.clear();

}

MarkSweepCollector::SlabData *MarkSweepCollector::getSlabData(void *ptr) {
  void *aligned = MemoryManager::slabAlignedPtr(ptr);
  auto ret = slabLookup.find(aligned);
  if(ret == slabLookup.end()) {
    return NULL;
  }
  return m_slabs.data() + ret->second;
}

void MarkSweepCollector::markReachable(void *ptr) {
  MarkSweepCollector::SlabData *sd = getSlabData(ptr);
  if(sd) {
    size_t blockId = MemoryManager::getBlockId(sd->slab, ptr);
    FTRACE(3, "Marked {} as reachable (block {})\n", ptr, blockId);

    sd->usedBlocks.set(blockId);
  }
}

bool MarkSweepCollector::isBlockReachable(void *ptr) {
  MarkSweepCollector::SlabData *sd = getSlabData(ptr);
  if(sd) {
    size_t blockId = MemoryManager::getBlockId(sd->slab, ptr);
    return sd->usedBlocks[blockId];
  }
  return false;
}

void MarkSweepCollector::markDestructable(ObjectData const *obj) {
  destructable.push_back(obj);
  FTRACE(3, "Marked {} as destructable\n", obj);
}

void MarkSweepCollector::markObjectLive(void *ptr, DataType t) {
  liveSet.insert(std::make_pair(ptr, t));
  allocTraces[ptr] = getStackTrace();
}

///////////////////////////////////////////////////////////////////////////////

}
