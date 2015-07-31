# Introduction #

**nih** is a simple library for performing _all kinds of basic graphics stuff_
efficiently that the author wouldn't want **not** to be **invented here** - and that will necessarily be such for anybody else but him. ;)
Truth is, this tiny little name just sounded good as a short, easy namespace.

Besides simple classes for manipulating small objects from linear algebra and
monte carlo and quasi-monte carlo sampling, it contains fast algorithms
for point cloud manipulation, hierarchy building and stuff like that.

As a library, it's supposed to be highly modular: one can simply include
a subset of its modules in its own project and everything should work
just as well (as long as dependencies are respected).

check out the
[Doxygen documentation](http://slash-sandbox.googlecode.com/svn/trunk/%20slash-sandbox/nih/docs/html/modules.html)

## External Dependencies ##

  * CUDA 4.1
  * Thrust
  * [back40computing](http://code.google.com/p/back40computing)

## k-d Trees ##

The library offers primitives for fast k-d tree creation over point sets,
based on a Morton code construction similar to the one used in LBVH.<br>
The following table reports sorting rates (in milliseconds)for different sets of random 3-d points, on a GTX480:<br>
<br>
<img src='http://slash-sandbox.googlecode.com/svn/trunk/%20slash-sandbox/nih/images/kd-chart.jpg' />

<h2>Bounding Volume Hierarchies</h2>

The library supports several BVH builders:<br>
<br>
<h3>LBVH</h3>

Linear Bounding Volume Hierarchies are a kind of BVHs built sorting points<br>
over the Morton space-filling curve, and using middle spatial splits to<br>
separate them into clusters.<br>
The resulting hierarchies are not always very tight, but this is probably the fastest known method for building BVHs.<br>
The implementation offered by <b>nih</b> is based on the algorithms described in:<br>
<a href='http://research.nvidia.com/publication/simpler-and-faster-hlbvh-work-queues'>http://research.nvidia.com/publication/simpler-and-faster-hlbvh-work-queues</a>

The following table reports sorting rates (in milliseconds) for different sets of random 3-d points, on a GTX480:<br>
<br>
<img src='http://slash-sandbox.googlecode.com/svn/trunk/%20slash-sandbox/nih/images/lbvh-chart.jpg' />

<h3>Greedy SAH Builder</h3>

The library offers a novel massively parallel implementation of the standard<br>
full Surface Area Heuristic-based BVH construction technique - a method that creates very high quality trees by carefully clustering primitives so as to minimize their surface area.<br>
<br>
<h3>Binned SAH Builder</h3>

This is a middle-quality BVH builder, using the binned SAH algorithm<br>
described in:<br>
<a href='http://research.nvidia.com/publication/simpler-and-faster-hlbvh-work-queues'>http://research.nvidia.com/publication/simpler-and-faster-hlbvh-work-queues</a>

The following table reports sorting rates (in milliseconds) for different sets of random 3-d bboxes, on a GTX480:<br>
<br>
<img src='http://slash-sandbox.googlecode.com/svn/trunk/%20slash-sandbox/nih/images/binned-sah-chart.jpg' />

The current builder performs binning along all 3 axes. Doing it along the largest<br>
axis only would be roughly 3 times faster while producing slightly worse trees.