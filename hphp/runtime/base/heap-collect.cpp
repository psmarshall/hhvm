/*
   +----------------------------------------------------------------------+
   | HipHop for PHP                                                       |
   +----------------------------------------------------------------------+
   | Copyright (c) 2010-2015 Facebook, Inc. (http://www.facebook.com)     |
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
#include "hphp/runtime/base/req-containers.h"
#include "hphp/runtime/base/mixed-array-defs.h"
#include "hphp/runtime/base/memory-manager-defs.h"
#include "hphp/runtime/base/heap-scan.h"
#include "hphp/runtime/base/thread-info.h"
#include "hphp/util/alloc.h"
#include "hphp/util/trace.h"

#include <algorithm>
#include <iterator>
#include <vector>
#include <folly/Range.h>

namespace HPHP {
TRACE_SET_MOD(mm);
using HK = HeaderKind;

namespace {

struct Counter {
  size_t count{0};
  size_t bytes{0};
  void operator+=(size_t n) {
    bytes += n;
    count++;
  }
};

struct Marker {
  explicit Marker(BigHeap* heap) : heap(heap) {}
  void init();
  void trace();
  void sweep(std::vector<std::pair<Header*, std::size_t>>& allocs);

  // mark exact pointers
  void operator()(const StringData*);
  void operator()(const ArrayData*);
  void operator()(const ObjectData*);
  void operator()(const ResourceData*);
  void operator()(const ResourceHdr*);
  void operator()(const RefData*);
  void operator()(const TypedValue&);
  void operator()(const TypedValueAux& v) { (*this)(*(const TypedValue*)&v); }
  void operator()(const NameValueTable*);
  void operator()(const VarEnv*);
  void operator()(const RequestEventHandler*);

  // mark ambiguous pointers in the range [start,start+len)
  void operator()(const void* start, size_t len);

  // classes containing exact pointers
  void operator()(const String&);
  void operator()(const Array&);
  void operator()(const ArrayNoDtor&);
  void operator()(const Object&);
  void operator()(const Resource&);
  void operator()(const Variant&);
  void operator()(const StringBuffer&);
  void operator()(const NameValueTable&);
  void operator()(const AsioContext& p) { scanner().scan(p, *this); }
  void operator()(const VarEnv& venv) { (*this)(&venv); }

  template<class T> void operator()(const req::ptr<T>& p) {
    (*this)(p.get());
  }
  template<class T> void operator()(const req::vector<T>& c) {
    for (auto& e : c) (*this)(e);
  }
  template<class T> void operator()(const req::set<T>& c) {
    for (auto& e : c) (*this)(e);
  }
  template<class T,class U> void operator()(const std::pair<T,U>& p) {
    (*this)(p.first);
    (*this)(p.second);
  }
  template<class T,class U,class V,class W>
  void operator()(const req::hash_map<T,U,V,W>& c) {
    for (auto& e : c) (*this)(e); // each element is pair<T,U>
  }

  template <typename T>
  void operator()(const LowPtr<T>& p) {
    (*this)(p.get());
  }

  void operator()(const ArrayIter& iter) {
    scan(iter, *this);
  }
  void operator()(const MArrayIter& iter) {
    scan(iter, *this);
  }

  // TODO: these need to be implemented.
  void operator()(const ActRec&) { }
  void operator()(const Stack&) { }

  void operator()(const RequestEventHandler& h) { (*this)(&h); }

  // TODO (6512343): this needs to be hooked into scan methods for Extensions.
  void operator()(const Extension&) { }

  // Explicitly ignored field types.
  void operator()(const LowPtr<Class>&) {}
  void operator()(const Func*) {}
  void operator()(const Class*) {}
  void operator()(const Unit*) {}
  void operator()(const std::string&) {}
  void operator()(int) {}

private:
  template<class T> static bool counted(T* p) {
    return p && p->isRefCounted();
  }
  bool mark(const void*);
  bool inRds(const void* vp) {
    auto p = reinterpret_cast<const char*>(vp);
    return p >= rds_.begin() && p < rds_.end();
  }
  template<class T> void enqueue(const T* p) {
    auto h = reinterpret_cast<const Header*>(p);
    assert(h &&
           h->kind() <= HK::BigMalloc &&
           h->kind() != HK::ResumableFrame &&
           h->kind() != HK::NativeData);
    work_.push_back(h);
  }

private:
  BigHeap* heap;
  std::vector<const Header*> work_;
  folly::Range<const char*> rds_; // full mmap'd rds section.
  Counter total_;        // bytes allocated in heap
};

// mark the object at p, return true if first time.
bool Marker::mark(const void* p) {
  assert(p && (heap->containsBig(p) || heap->testMapBit(p)));
  auto h = static_cast<const Header*>(p);
  assert(h->kind() <= HK::BigMalloc && h->kind() != HK::ResumableObj);
  auto first = !h->hdr_.mark;
  h->hdr_.mark = true;
  return first;
}

// Utility to just extract the kind field from an arbitrary Header ptr.
inline DEBUG_ONLY HeaderKind kind(const void* p) {
  return static_cast<const Header*>(p)->kind();
}

void Marker::operator()(const ObjectData* p) {
  if (!p) return;
  assert(isObjectKind(p->headerKind()));
  if (p->getAttribute(ObjectData::HasNativeData)) {
    // HNI style native object; mark the NativeNode header, queue the object.
    // [NativeNode][NativeData][ObjectData][props] is one allocation.
    // For generators -
    // [NativeNode][locals][Resumable][GeneratorData][ObjectData]
    auto h = Native::getNativeNode(p, p->getVMClass()->getNativeDataInfo());
    assert(h->hdr.kind == HK::NativeData);
    if (mark(h)) {
      enqueue(p);
    }
  } else if (p->headerKind() == HK::ResumableObj) {
    // Resumable object, prefixed by a ResumableNode header, which is what
    // we need to mark.
    // [ResumableNode][locals][Resumable][ObjectData<ResumableObj>]
    auto r = Resumable::FromObj(p);
    auto frame = reinterpret_cast<const TypedValue*>(r) -
                 r->actRec()->func()->numSlotsInFrame();
    auto node = reinterpret_cast<const ResumableNode*>(frame) - 1;
    assert(node->hdr.kind == HK::ResumableFrame);
    FTRACE(2, "!! mark resumable at {}, node={}\n", p, node);
    if (mark(node)) {
      // mark the ResumableFrame prefix, but enqueue the ObjectData* to scan
      enqueue(p);
    }
  } else {
    // Ordinary non-builtin object subclass, or IDL-style native object.
    if (mark(p)) {
      enqueue(p);
    }
  }
}

void Marker::operator()(const ResourceHdr* p) {
  if (p && mark(p)) {
    assert(kind(p) == HK::Resource);
    enqueue(p);
  }
}

void Marker::operator()(const ResourceData* r) {
  if (r && mark(r->hdr())) {
    assert(kind(r->hdr()) == HK::Resource);
    enqueue(r->hdr());
  }
}

// ArrayData objects could be static
void Marker::operator()(const ArrayData* p) {
  if (p && counted(p) && mark(p)) {
    assert(isArrayKind(kind(p)));
    enqueue(p);
  }
}

// RefData objects contain at most one ptr, scan it eagerly.
void Marker::operator()(const RefData* p) {
  if (!p) return;
  if (inRds(p)) {
    // p is a static local, initialized by RefData::initInRDS().
    // we already scanned p's body as part of scanning RDS.
    return;
  }
  if (mark(p)) {
    assert(kind(p) == HK::Ref);
    enqueue(p);
  }
}

// The only thing interesting in a string is a possible APCString*,
// which is not a request-local allocation.
void Marker::operator()(const StringData* p) {
  if (p && counted(p)) {
    assert(kind(p) == HK::String);
    mark(p);
  }
}

// NVTs live inside VarEnv, and GlobalsArray has an interior ptr to one.
// ignore the interior pointer; NVT should be scanned by VarEnv::scan.
void Marker::operator()(const NameValueTable* p) {}

// VarEnvs are allocated with req::make, so they aren't first-class heap
// objects. assume a VarEnv* is a unique ptr, and scan it eagerly.
void Marker::operator()(const VarEnv* p) {
  if (p) p->scan(*this);
}

void Marker::operator()(const RequestEventHandler* p) {
  p->scan(*this);
}

void Marker::operator()(const String& p)    { (*this)(p.get()); }
void Marker::operator()(const Array& p)     { (*this)(p.get()); }
void Marker::operator()(const ArrayNoDtor& p) { (*this)(p.arr()); }
void Marker::operator()(const Object& p)    { (*this)(p.get()); }
void Marker::operator()(const Resource& p)  { (*this)(p.hdr()); }
void Marker::operator()(const Variant& p)   { (*this)(*p.asTypedValue()); }

void Marker::operator()(const StringBuffer& p) { p.scan(*this); }
void Marker::operator()(const NameValueTable& p) { p.scan(*this); }

// mark a TypedValue or TypedValueAux. taking tv by value would exclude aux.
void Marker::operator()(const TypedValue& tv) {
  switch (tv.m_type) {
    case KindOfString:    return (*this)(tv.m_data.pstr);
    case KindOfArray:     return (*this)(tv.m_data.parr);
    case KindOfObject:    return (*this)(tv.m_data.pobj);
    case KindOfResource:  return (*this)(tv.m_data.pres);
    case KindOfRef:       return (*this)(tv.m_data.pref);
    case KindOfUninit:
    case KindOfNull:
    case KindOfBoolean:
    case KindOfInt64:
    case KindOfDouble:
    case KindOfStaticString:
    case KindOfClass: // only in eval stack
      return;
  }
}

// mark ambigous pointers in the range [start,start+len). If the start or
// end is a partial word, don't scan that word.
void FOLLY_DISABLE_ADDRESS_SANITIZER
Marker::operator()(const void* start, size_t len) {
  constexpr uintptr_t M{7}; // word size - 1
  auto s = (char**)((uintptr_t(start) + M) & ~M); // round up
  auto e = (char**)((uintptr_t(start) + len) & ~M); // round down
  for (; s < e; s++) {
    auto p = *s;
    if (MemoryManager::align(p) != p) {
      TRACE(2, "!! Skipping non-aligned ambiguous ptr %p\n", p);
      continue;
    }
    auto real = heap->containsBig(p) || (heap->contains(p) && heap->testMapBit(p));
    if (!real) {
      continue;
    }
    auto h = (Header*)p;
    // mark p if it's an interesting kind. since we have metadata for it,
    // it must have a valid header.
    h->hdr_.cmark = true;
    if (!mark(h)) continue; // skip if already marked.
    switch (h->kind()) {
      case HK::Apc:
      case HK::Globals:
      case HK::Proxy:
      case HK::Ref:
      case HK::Resource:
      case HK::Packed:
      case HK::Struct:
      case HK::Mixed:
      case HK::Empty:
      case HK::SmallMalloc:
      case HK::BigMalloc:
        enqueue(h);
        break;
      case HK::Object:
      case HK::AwaitAllWH:
      case HK::Vector:
      case HK::Map:
      case HK::Set:
      case HK::Pair:
      case HK::ImmVector:
      case HK::ImmMap:
      case HK::ImmSet:
        // Object kinds. None of these should have native-data, because if they
        // do, the mapped header should be for the NativeData prefix.
        assert(!h->obj_.getAttribute(ObjectData::HasNativeData));
        enqueue(h);
        break;
      case HK::ResumableFrame:
        enqueue(h->resumableObj());
        break;
      case HK::NativeData:
        enqueue(h->nativeObj());
        break;
      case HK::String:
        // nothing to queue since strings don't have pointers
        break;
      case HK::ResumableObj:
      case HK::BigObj:
      case HK::Free:
      case HK::Hole:
        // None of these kinds should be encountered because they're either not
        // interesting to begin with, or are mapped to different headers, so we
        // shouldn't get these from the pointer map.
        always_assert(false && "bad header kind");
        break;
    }
  }
}

// initially parse the heap to find valid objects and initialize metadata.
// Certain objects can have count==0
// * StringData owned by StringBuffer
// * ArrayData owned by ArrayInit
// * Object ctors allocating memory in ctor (while count still==0).
void Marker::init() {
  rds_ = folly::Range<const char*>((char*)rds::header(),
                                   RuntimeOption::EvalJitTargetCacheSize);

  MM().iterate([&](Header* h) {
    h->hdr_.mark = h->hdr_.cmark = false;
    switch (h->kind()) {
      case HK::Apc:
      case HK::Globals:
      case HK::Proxy:
      case HK::Packed:
      case HK::Mixed:
      case HK::Struct:
      case HK::Empty:
      case HK::String:
        total_ += h->size();
        break;
      case HK::Ref:
        // EZC non-ref refdatas sometimes have count==0
        // assert(h->hdr_.count > 0 || !h->ref_.zIsRef());
        total_ += h->size();
        break;
      case HK::Resource:
        // ZendNormalResourceData objects sometimes never incref'd
        // TODO: t5969922, t6545412 might be a real bug.
        total_ += h->size();
        break;
      case HK::Object:
      case HK::Vector:
      case HK::Map:
      case HK::Set:
      case HK::Pair:
      case HK::ImmVector:
      case HK::ImmMap:
      case HK::ImmSet:
      case HK::AwaitAllWH:
        // count==0 can be witnessed, see above
        total_ += h->size();
        if (h->obj_.getAttribute(ObjectData::HasNativeData)) {
          // Objects with native-data shouldn't be encountered on their own
          // because they should be prefixed by a NativeData allocation.
          assert(false && "object with native-data");
        }
        break;
      case HK::ResumableFrame: {
        // Pointers to either the frame or the object will be mapped to the
        // frame.
        total_ += h->size();
        auto obj = reinterpret_cast<const Header*>(h->resumableObj());
        obj->hdr_.mark = obj->hdr_.cmark = false;
        FTRACE(2, "Resumable frame in init: h={}, obj={}\n", h, obj);
        break;
      }
      case HK::NativeData: {
        // Pointers to either the native data or the object will be mapped to
        // the native data.
        total_ += h->size();
        auto obj = reinterpret_cast<const Header*>(h->nativeObj());
        obj->hdr_.mark = obj->hdr_.cmark = false;
        break;
      }
      case HK::SmallMalloc:
      case HK::BigMalloc:
        total_ += h->size();
        break;
      case HK::Free:
        break;
      case HK::ResumableObj:
        // These shouldn't be encountered on their own, they should always be
        // prefixed by a ResumableFrame allocation.
        assert(false && "Get outta here");
        break;
      case HK::Hole:
      case HK::BigObj:
        assert(false && "Get outta here");
        break;
    }
  });

  //clear marks for immix lines
  MM().forEachLine([&](void* line, uint8_t& markByte) {
    markByte = 0;
  });
}

void Marker::trace() {
  scanRoots(*this);
  while (!work_.empty()) {
    auto h = work_.back();
    work_.pop_back();
    scanHeader(h, *this);
  }
  // mark all immix lines containing SmallMalloc headers
  // because we don't actually free them yet
  // We don't need to do this for BigMalloc header because they
  // aren't stored in immix lines/blocks, so they won't be
  // accidentally freed
  // Use iterateSlabs because it ignores bigs, it will be faster
  MM().iterateSlabs([&](Header* h, const live_map::reference live) {
    if (h->kind() == HK::SmallMalloc) {
      // mark line
      TRACE(3, "Marking line for SmallMalloc at %p\n", h);
      if (h->size() > kLineSize) {
        TRACE(3, "Marking multiple lines for SmallMalloc at %p\n", h);
        MM().markLinesForMedium(h, h->size());
      } else {
        MM().markLineForSmall(h);
      }
    }
  });
}

// check that headers have a "sensible" state during sweeping.
DEBUG_ONLY bool check_sweep_header(const Header* h) {
  assert(!h->hdr_.cmark || h->hdr_.mark); // cmark implies mark
  switch (h->kind()) {
    case HK::Packed:
    case HK::Struct:
    case HK::Mixed:
    case HK::Empty:
    case HK::Apc:
    case HK::Globals:
    case HK::Proxy:
    case HK::String:
    case HK::Resource:
    case HK::Ref:
      // ordinary counted objects
      break;
    case HK::Object:
    case HK::Vector:
    case HK::Map:
    case HK::Set:
    case HK::Pair:
    case HK::ImmVector:
    case HK::ImmMap:
    case HK::ImmSet:
    case HK::AwaitAllWH:
      // objects; should not have native-data
      assert(!h->obj_.getAttribute(ObjectData::HasNativeData));
      break;
    case HK::ResumableFrame:
    case HK::NativeData:
      // not counted but marked when embedded object is marked
      break;
    case HK::SmallMalloc:
    case HK::BigMalloc:
      // not counted but can be marked.
      break;
    case HK::Free:
      // free memory; these should not be marked.
      assert(!h->hdr_.mark);
      break;
    case HK::ResumableObj:
    case HK::BigObj:
    case HK::Hole:
      // These should never be encountered because they don't represent
      // independent allocations.
      assert(false && "invalid header kind");
      break;
  }
  return true;
}

// another pass through the heap now that everything is marked.
void Marker::sweep(std::vector<std::pair<Header*, std::size_t>>& allocs) {
  Counter marked, ambig, freed;
  std::vector<Header*> reaped;
  std::vector<std::pair<Header*, std::size_t>> survivors;

  auto& mm = MM();

  //TODO this doesn't free bigs, should be OK?

  // recreate live-map using mark info
  mm.iterateSlabs([&](Header* h, live_map::reference live) {
    if (h->hdr_.mark) {
      // mark our line
      if (h->size() <= kLineSize) {
        mm.markLineForSmall(h);
      } else if (h->size() <= kMaxMediumSize) {
        mm.markLinesForMedium(h, h->size());
      }
      // live bit is true, don't need to change it
      survivors.emplace_back(h, h->size()); // object lives, don't do anything to it
      return;
    }
    // object dies
    live = false; // flick off our live bit
    switch (h->kind()) {
      case HK::Apc:
      case HK::String:
      case HK::Object:
      case HK::ResumableFrame:
      case HK::NativeData:
        freed += h->size();
        reaped.push_back(h);
        break;
      case HK::Packed:
      case HK::Struct:
      case HK::Mixed:
      case HK::Empty:
      case HK::Globals:
      case HK::Proxy:
      case HK::Resource:
      case HK::Ref:
      case HK::AwaitAllWH:
      case HK::Vector:
      case HK::Map:
      case HK::Set:
      case HK::Pair:
      case HK::ImmVector:
      case HK::ImmMap:
      case HK::ImmSet:
        freed += h->size();
        break;
      case HK::SmallMalloc:
        live = true; // need to keep these live 
        break;
      case HK::BigMalloc:
        // Don't free malloc-ed allocations even if they're not reachable.
        break;
      case HK::Free:
        assert(false && "Free shouldn't be in Immix heap");
        break;
      case HK::Hole:
      case HK::BigObj:
      case HK::ResumableObj:
        assert(false && "get outta here");
        break;
    }
  });
  
  FTRACE(2, "allocations before sweep: {}, removed: {}, remaining: {}\n",
    allocs.size(), allocs.size() - survivors.size(), survivors.size());

  allocs = survivors;

  for (const auto& pair : survivors) {
    bool alive = heap->testMapBit(pair.first);
    if (!alive) {
      FTRACE(2, "!! survivor not marked in live-map ptr={}\n", pair.first);
    }
    assert(alive);
  }

  // It's safe to free unreachable objects.
  // None of these actually *free* anything but they may run
  // finalization logic
  for (auto h : reaped) {
    bool alive = heap->testMapBit(h);
    if (alive) {
      FTRACE(2, "!! dead thing marked in live-map ptr={}\n", h);
    }
    assert(!alive);
    if (h->kind() == HK::Apc) {
      h->apc_.reap(); // calls smart_free() and smartFreeSize()
    } else if (h->kind() == HK::String) {
      h->str_.release(); // no destructor can run, so release() is safe.
    } else if (auto obj = h->obj()) {
      if (obj->getAttribute(ObjectData::HasDynPropArr)) {
        g_context->dynPropTable.erase(obj);
      }
      // mm.objFree(h, h->size());
    }
  }

  // immix free lines
  if (debug) {
    uint32_t kept = 0, total = 0, implicit = 0;
    uint8_t prevMarkByte = 0; // deals with implicit marking
    mm.forEachLine([&](void* line, uint8_t& markByte) {
      total++;
      if (markByte == 0) {
        if (prevMarkByte == 0) {
          TRACE(4, "line freed %p\n", line);
          // doesn't matter we are putting trash in the heap
          // because it doesn't need to be parseable
          memset(line, kSmallFreeFill, kLineSize);
        } else {
          implicit++;
        }
      } else {
        FTRACE(4, "line kept {}, markByte: {}, prevMarkByte: {}\n",
          line, markByte, prevMarkByte);
        kept++;
      }
      prevMarkByte = markByte;
    });
    FTRACE(2, "#lines at sweep: {}, marked: {}, implicitly marked: {}, wiped: {}\n",
      total, kept, implicit, total - kept);
  }

  mm.freeUnusedBlocks();

  // reset to start allocating into blocks from the start again
  mm.goToFirstRecyclableBlock();

  TRACE(1, "sweep tot %lu(%lu) mk %lu(%lu) amb %lu(%lu) free %lu(%lu)\n",
        total_.count, total_.bytes,
        marked.count, marked.bytes,
        ambig.count, ambig.bytes,
        freed.count, freed.bytes);
}
}

void MemoryManager::collect() {
  TRACE(1, "MemoryManager::collect() called\n");
  if (!RuntimeOption::EvalEnableGC || empty()) return;

  m_heap.dumpMapBits();

  for (const auto& pair : m_heap_allocations) {
    bool alive = m_heap.testMapBit(pair.first);
    if (!alive) {
      FTRACE(2, "!! heap_ptr={}\n", pair.first);
    }
    assert(alive);
  }

  Marker mkr(&m_heap);
  mkr.init();
  mkr.trace();
  mkr.sweep(m_heap_allocations);
}

}
