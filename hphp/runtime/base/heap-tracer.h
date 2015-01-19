#ifndef incl_HPHP_HEAP_TRACER_H_
#define incl_HPHP_HEAP_TRACER_H_

#include "hphp/runtime/base/base-includes.h"
#include <functional>
#include <queue>
#include <unordered_set>

namespace HPHP {

///////////////////////////////////////////////////////////////////////////////

struct SearchNode {
  TypedValue current;
  TypedValue parent;
};

typedef std::function<void(const SearchNode)> NodeHandleFunc;

class HeapTracer {
public:
  HeapTracer() {}
  ~HeapTracer() {}

  void traceHeap(NodeHandleFunc nodeFun);

private:
  void traceStackFrame(const ActRec *fp, int offset, const TypedValue *ftop);

  void reset();

  void markReachable(void *ptr);

  bool isReachable(void *ptr);

  std::unordered_set<const void *> marked;
  std::queue<SearchNode> m_searchQ;
};

///////////////////////////////////////////////////////////////////////////////
}

#endif
